# app.py
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

# =============================================================
# Student Tutoring and Resource Finder — Streamlit App
# Tabs:
#  - 🎓 Tutor (Personalized Learning Tutor chatbot)
#  - 🔎 Find Resources (OSM live search + local DB + map)
#  - 🆘 Emergency Assistance (contacts + AI guidance)
# =============================================================

st.set_page_config(
    page_title="Student Tutoring and Resource Finder",
    page_icon="favicon.png",
    layout="wide"
) 
# ------------------------------
# OpenAI client (put your key in Streamlit Secrets)
# ------------------------------
try:
    client = OpenAI(api_key="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")
except Exception:
    client = OpenAI()  # falls back to env var OPENAI_API_KEY

# ------------------------------
# Defaults
# ------------------------------
DB_PATH = "resources_database.xlsx"  # keep in ROOT

# ------------------------------
# Utilities
# ------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    alias_map = {
        "latitude": "lat", "longitude": "lon",
        "phone number": "phone", "telephone": "phone",
        "url": "website", "site": "website",
        "resource type": "type", "category": "type",
    }
    for old, new in alias_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    for col in ["name", "type", "address", "phone", "website", "lat", "lon"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

@st.cache_data(show_spinner=False)
def load_default_database(path: str = DB_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["name","type","address","phone","website","lat","lon"])
    df = pd.read_excel(path)  # requires openpyxl
    return _normalize_columns(df)

# ------------------------------
# RAG: build embeddings from a DataFrame
# ------------------------------
@st.cache_data(show_spinner=False)
def build_kb_embeddings(df: pd.DataFrame):
    if df.empty:
        return None, [], df

    def row_to_text(rec: dict) -> str:
        fields = []
        for key in ["name","type","address","city","state","phone","website","eligibility","hours"]:
            if key in rec and str(rec.get(key, "")).strip():
                fields.append(f"{key}: {rec[key]}")
        return " | ".join(fields)

    rows = df.fillna("").to_dict(orient="records")
    texts = [row_to_text(r) for r in rows]

    try:
        embeds = []
        batch_size = 96
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            resp = client.embeddings.create(model="text-embedding-3-small", input=chunk)
            embeds.extend([e.embedding for e in resp.data])
        E = np.array(embeds, dtype=np.float32)
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        return E, texts, df.reset_index(drop=True)
    except Exception as e:
        st.warning(f"Embedding build failed: {e}")
        return None, texts, df.reset_index(drop=True)

def retrieve_from_kb(query: str, top_k: int = 5) -> Optional[str]:
    E = st.session_state.get("KB_EMBEDS")
    kb_df = st.session_state.get("KB_DF")
    if E is None or kb_df is None or kb_df.empty:
        return None
    try:
        q_emb = client.embeddings.create(model="text-embedding-3-small", input=[query]).data[0].embedding
        import numpy as _np
        qv = _np.array(q_emb, dtype=_np.float32)
        qv /= (_np.linalg.norm(qv) + 1e-8)
        sims = E @ qv
        idx = _np.argsort(-sims)[:top_k]
        rows = []
        for i in idx:
            row = kb_df.iloc[int(i)].to_dict()
            summary = ", ".join(
                f"{k}: {v}" for k, v in row.items()
                if str(v).strip() and k in ["name","type","address","phone","website","hours","eligibility","lat","lon"]
            )
            rows.append(f"- {summary}")
        return "\n".join(rows) if rows else None
    except Exception as e:
        st.caption(f"RAG lookup skipped: {e}")
        return None

# ------------------------------
# OpenAI chat helper
# ------------------------------
def ai_complete(messages: List[Dict], temperature: float = 0.2, max_tokens: int = 500, system_prefix: Optional[str] = None) -> str:
    try:
        final_messages = []
        if system_prefix:
            final_messages.append({"role": "system", "content": system_prefix})
        final_messages.extend(messages)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=final_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI unavailable: {e})"

def ask_with_rag(user_msg: str) -> str:
    rag_blob = retrieve_from_kb(user_msg, top_k=5)
    system_prefix = None
    if rag_blob:
        system_prefix = (
            "Use the following local resource database entries if relevant. "
            "Only include items that match the user's need, and provide phone numbers and addresses when available.\n\n"
            f"LOCAL_KB:\n{rag_blob}\n\n"
        )
    return ai_complete([{"role": "user", "content": user_msg}], system_prefix=system_prefix)

# ------------------------------
# OSM Live search helpers
# ------------------------------
OSM_FILTERS = {
    "Food Bank": [
        '[amenity="food_bank"]',
        '[amenity="social_facility"][social_facility="food_bank"]',
    ],
    "Shelter": [
        '[amenity="shelter"]',
        '[amenity="social_facility"][social_facility="shelter"]',
        '[amenity="social_facility"]["social_facility:for"="homeless"]',
    ],
    "Healthcare Clinic": [
        '[amenity="clinic"]',
        '[amenity="doctors"]',
        '[amenity="hospital"]',
    ],
    "Job Training": [
        '[office="employment_agency"]',
        '[amenity="social_facility"][social_facility="outreach"]',
    ],
    "Affordable Housing": [
        '[social_facility="group_home"]',
        '[building="apartments"]["operator:type"="public"]',
    ],
    "Legal Aid": [
        '[office="lawyer"]',
        '[amenity="social_facility"][social_facility="outreach"]',
    ],
}

@st.cache_data(show_spinner=False)
def geocode_address(address: str):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1},
            headers={"User-Agent": "streamlit-resource-locator/1.0"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        item = data[0]
        return float(item["lat"]), float(item["lon"]), item.get("display_name", address)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def overpass_query(lat: float, lon: float, radius_km: float, resource_types: List[str], limit: int = 200):
    radius_m = max(1, int(radius_km * 1000))
    parts: List[str] = []
    for rt in resource_types:
        for f in OSM_FILTERS.get(rt, []):
            parts.append(f'node{f}(around:{radius_m},{lat},{lon});')
            parts.append(f'way{f}(around:{radius_m},{lat},{lon});')
            parts.append(f'relation{f}(around:{radius_m},{lat},{lon});')
    if not parts:
        return {"elements": []}
    q = f"""[out:json][timeout:30];
(
{chr(10).join(parts)}
);
out tags center {int(limit)};
"""
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    last_err = None
    for url in endpoints:
        try:
            r = requests.post(
                url,
                data={"data": q},
                headers={"User-Agent": "streamlit-resource-locator/1.0"},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
    st.error(f"Overpass query failed: {last_err}")
    with st.expander("Show Overpass QL (copy into overpass-turbo.eu to debug)"):
        st.code(q, language="sql")
    return {"elements": []}

@st.cache_data(show_spinner=False)
def parse_overpass(json_obj, default_type: str = "Resource") -> pd.DataFrame:
    rows: List[Dict] = []
    for el in json_obj.get("elements", []):
        tags = el.get("tags", {}) or {}
        name = tags.get("name") or tags.get("operator") or default_type
        lat = el.get("lat") or (el.get("center", {}) or {}).get("lat")
        lon = el.get("lon") or (el.get("center", {}) or {}).get("lon")
        addr = ", ".join(filter(None, [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
            tags.get("addr:city"),
            tags.get("addr:state"),
            tags.get("addr:postcode"),
        ])) or tags.get("addr:full") or ""
        phone = tags.get("phone") or tags.get("contact:phone") or ""
        website = tags.get("website") or tags.get("contact:website") or ""
        rtype = default_type
        amenity = tags.get("amenity")
        if amenity in ("clinic", "doctors", "hospital"):
            rtype = "Healthcare Clinic"
        elif amenity in ("shelter",):
            rtype = "Shelter"
        elif amenity in ("food_bank",):
            rtype = "Food Bank"
        if tags.get("office") == "employment_agency":
            rtype = "Job Training"
        if tags.get("social_facility") == "food_bank":
            rtype = "Food Bank"
        if tags.get("social_facility") == "shelter":
            rtype = "Shelter"
        rows.append({
            "name": name, "type": rtype, "address": addr, "phone": phone,
            "website": website, "lat": lat, "lon": lon,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return df

# ------------------------------
# Emergency contacts (static)
# ------------------------------
EMERGENCY_CONTACTS = {
    "Emergency (USA)": {"phone": "911", "desc": "Immediate danger or medical emergency"},
    "988 Suicide & Crisis Lifeline": {"phone": "988", "desc": "24/7 mental health crisis support"},
    "National Domestic Violence Hotline": {"phone": "1-800-799-7233", "desc": "Confidential DV support"},
    "SAMHSA Helpline": {"phone": "1-800-662-4357", "desc": "Substance use & mental health referrals"},
    "Legal Aid (Generic)": {"phone": "1-800-555-1234", "desc": "Legal info & referrals (placeholder)"},
}

st.sidebar.header("🚨 Emergency Assistance")
st.sidebar.caption("If you are in immediate danger, call local emergency services.")
for name, info in EMERGENCY_CONTACTS.items():
    st.sidebar.markdown(f"**{name}**: {info['phone']}")
    st.sidebar.caption(info["desc"])

# ------------------------------
# Initialize DB + embeddings (default file)
# ------------------------------
if "KB_READY" not in st.session_state:
    db_df = load_default_database(DB_PATH)  # reads resources_database.xlsx if present
    E, texts, kb_df = build_kb_embeddings(db_df)
    st.session_state["KB_EMBEDS"] = E
    st.session_state["KB_TEXTS"]  = texts
    st.session_state["KB_DF"]     = kb_df
    st.session_state["KB_READY"]  = True

# ------------------------------
# Big centered title (site title)
# ------------------------------
st.markdown(
    "<h1 style='text-align:center;margin:0;'>Student Tutoring and Resource Finder</h1>",
    unsafe_allow_html=True,
)
st.write("")  # spacer

# ------------------------------
# Tabs (Tutor FIRST)
# ------------------------------
TAB_TUTOR, TAB_HOME, TAB_EMERGENCY = st.tabs(["🎓 Tutor", "🔎 Find Resources", "🆘 Emergency Assistance"])

with TAB_TUTOR:
    st.subheader("🤖 Personalized Learning Tutor")
    st.markdown(
        "Welcome! I'm your AI tutor designed for students. Tell me your grade and subjects, "
        "and I'll create custom lessons, quizzes, and explanations. This is free and tailored for accessible learning."
    )

    def build_tutor_system_prompt(_grade: str, _subjects: List[str]) -> str:
        s = ", ".join(_subjects) if _subjects else "general studies"
        return (
            f"You are a friendly, patient tutor for a {_grade.lower()} student in {s}. "
            "Keep explanations simple, engaging, and step-by-step. Use examples from everyday life. "
            "For lessons: Break into short sections with objectives. "
            "For quizzes: Create 3-5 multiple-choice or short-answer questions with answers explained. "
            "For explanations: Use analogies and describe simple visuals. "
            "Suggest free resources like Khan Academy videos when relevant. "
            "Always encourage the student and ask what they'd like next."
        )

    col1, col2 = st.columns([1, 2])
    with col1:
        grade = st.selectbox(
            "Your Grade Level",
            [
                "Elementary (K-1)", "Elementary (2)", "Elementary (3)", "Elementary (4)", "Elementary (5)",
                "Middle School (6)", "Middle School (7)", "Middle School (8)",
                "High School (9-12)", "College Prep",
            ],
            key="tutor_grade",
        )
        subjects = st.multiselect(
            "Subjects (select or type)",
            ["Math", "Science", "English", "History", "Other"],
            default=[],
            key="tutor_subjects",
        )
        if "Other" in subjects:
            other_subject = st.text_input("Specify 'Other' subject:", key="tutor_other_subject")
            if other_subject:
                subjects = [s for s in subjects if s != "Other"] + [other_subject]
        st.info(f"Profile: {grade} | Subjects: {', '.join(subjects) if subjects else 'None selected'}")

    profile_key = (grade, tuple(subjects))
    if st.session_state.get("_tutor_profile_key") != profile_key:
        st.session_state["_tutor_profile_key"] = profile_key
        st.session_state["tutor_messages"] = [
            {"role": "system", "content": build_tutor_system_prompt(grade, subjects)}
        ]

    for m in st.session_state.get("tutor_messages", [])[1:]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    tutor_prompt = st.chat_input(
        "Ask me anything about your subjects! E.g., 'Explain fractions' or 'Quiz me on algebra'",
        key="tutor_chat_input",
    )
    if tutor_prompt:
        st.session_state.tutor_messages.append({"role": "user", "content": tutor_prompt})
        with st.chat_message("user"):
            st.markdown(tutor_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking and preparing your lesson..."):
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.tutor_messages,
                    max_tokens=700,
                    temperature=0.7,
                )
                llm_response = resp.choices[0].message.content
                st.markdown(llm_response)
        st.session_state.tutor_messages.append({"role": "assistant", "content": llm_response})

with TAB_HOME:
    st.subheader("📍 Resource Locator")
    st.caption("Browse food banks, shelters, clinics, job training, and affordable housing.")

    st.markdown("### 🧩 Load your resource database (Excel/CSV)")
    up = st.file_uploader("Upload .xlsx, .xls, or .csv", type=["xlsx", "xls", "csv"], accept_multiple_files=False)
    sheet = None
    user_df = None
    if up is not None and up.name.lower().endswith((".xlsx", ".xls")):
        try:
            xls = pd.ExcelFile(up)
            sheet = st.selectbox("Select sheet", xls.sheet_names)
            user_df = pd.read_excel(xls, sheet_name=sheet)
        except Exception as e:
            st.warning(f"Could not enumerate/load sheets: {e}")
    elif up is not None:
        try:
            user_df = pd.read_csv(up)
        except Exception as e:
            st.warning(f"Could not read CSV: {e}")

    if user_df is not None:
        user_df = _normalize_columns(user_df)
        if user_df.empty:
            st.error("Uploaded file has no readable rows.")
        else:
            st.success(f"Loaded {len(user_df)} rows from {up.name}{' / ' + (sheet or '')}.")
            E, texts, kb_df = build_kb_embeddings(user_df)
            st.session_state["KB_EMBEDS"] = E
            st.session_state["KB_TEXTS"]  = texts
            st.session_state["KB_DF"]     = kb_df

    active_df = st.session_state.get("KB_DF", pd.DataFrame())

    st.markdown("#### 🔍 Search live data by address")
    address = st.text_input(
        "Enter an address or city",
        value="4400 Water Oak Rd, Charlotte, NC 28211",
        placeholder="e.g., 123 Main St, Charlotte NC",
        key="addr_input",
    )
    resource_choices = [
        "Food Bank", "Shelter", "Healthcare Clinic", "Job Training", "Affordable Housing", "Legal Aid",
    ]
    selected_types = st.multiselect(
        "Resource types",
        resource_choices,
        default=resource_choices,
        key="types_select",
    )
    radius_km = st.slider("Search radius (km)", 1, 50, 10)

    cache = st.session_state.setdefault("live_cache", {})
    cache_key = (address.strip(), tuple(sorted(selected_types)), int(radius_km))

    if address and selected_types:
        if cache_key in cache:
            live_df, gdisp = cache[cache_key]
        else:
            with st.spinner("Looking up address and nearby resources…"):
                gc = geocode_address(address)
                if not gc:
                    st.error("Could not geocode that address. Try another.")
                    live_df, gdisp = pd.DataFrame(), None
                else:
                    glat, glon, gdisp = gc
                    try:
                        resp = overpass_query(glat, glon, radius_km, selected_types)
                        live_df = parse_overpass(resp)
                    except Exception as e:
                        st.error(f"Overpass query failed: {e}")
                        live_df = pd.DataFrame()
            cache[cache_key] = (live_df, gdisp)

        if not live_df.empty:
            st.session_state["live_results"] = live_df
            st.success(f"Found {len(live_df)} places near {gdisp}.")
        else:
            st.warning("No results found in this area for the selected types.")
    else:
        st.info("Enter an address to search.")

    if "live_results" in st.session_state:
        live_df = st.session_state["live_results"]
        st.map(live_df.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]])
        st.dataframe(live_df[["name", "type", "address", "phone", "website"]], use_container_width=True)

    st.markdown("---")
    st.subheader("Active database view")
    type_choice = st.selectbox(
        "Filter list by type",
        ["All", "Food Bank", "Shelter", "Healthcare Clinic", "Job Training", "Affordable Housing", "Legal Aid"],
    )
    df_local = active_df.copy()
    if not df_local.empty and type_choice != "All":
        if "type" in df_local.columns:
            df_local = df_local[df_local["type"].str.lower().fillna("") == type_choice.lower()]
    if not df_local.empty and {"lat", "lon"}.issubset(df_local.columns):
        valid = df_local[["lat", "lon"]].dropna()
        if not valid.empty:
            st.map(valid.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]])
    st.dataframe(df_local, use_container_width=True)

with TAB_EMERGENCY:
    st.subheader("🆘 Emergency Assistance")
    st.caption("Crisis hotlines, shelters, and legal aid contacts.")

    st.markdown("### 📞 Crisis Hotlines (U.S.)")
    st.markdown(
        "- **Emergency:** 911"
        "- **988 Suicide & Crisis Lifeline:** 988"
        "- **National Domestic Violence Hotline:** 1-800-799-7233"
        "- **SAMHSA (Substance Use/Mental Health):** 1-800-662-4357"
    )

    st.markdown("### 🏠 Nearby Shelters & Food (use the search above on Home tab)")
    st.markdown("### ⚖️ Legal Aid (local directories vary)")
    st.caption("Use the live search with 'Legal Aid' type, or consult your county/state legal aid directory.")

    st.markdown("---")
    st.subheader("💬 Guidance Chat (OpenAI)")
    if "em_chat" not in st.session_state:
        st.session_state.em_chat = [
            {
                "role": "system",
                "content": (
                    "You are a calm, supportive guidance assistant inside a community resource app. "
                    "Be brief, practical, and trauma-informed. Encourage contacting local emergency services when risk is high. "
                    "Avoid collecting personal identifiers. Do not provide professional legal or medical advice; instead, point to qualified services."
                ),
            },
            {
                "role": "assistant",
                "content": "I'm here. Tell me what you need help with, and I'll guide you to nearby services.",
            },
        ]

    for msg in st.session_state.em_chat:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    u2 = st.chat_input("Type your question (avoid personal details)", key="em_chat_input")
    if u2:
        st.session_state.em_chat.append({"role": "user", "content": u2})
        reply = ask_with_rag(u2)
        st.session_state.em_chat.append({"role": "assistant", "content": reply})
        st.rerun()

