# app.py - Streamlit App for Personalized Learning Tutor Chatbot
# Deploy to Streamlit Community Cloud: Add to requirements.txt - streamlit, openai
# Secrets: Add OPENAI_API_KEY=your-key-here in Streamlit Cloud settings

import streamlit as st
from openai import OpenAI
import os

# Load OpenAI API key from secrets
#openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
openai_api_key = st.secrets.get("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# App title and description
st.title("🤖 Personalized Learning Tutor")
st.markdown("""
Welcome! I'm your AI tutor designed for students. Tell me your grade and subjects, and I'll create custom lessons, quizzes, and explanations.
This is free and tailored for accessible learning.
""")

# Sidebar for user profile (grade and subjects)
with st.sidebar:
    st.header("Your Profile")
    grade = st.selectbox("Your Grade Level", ["Elementary (K-1)", "Elementary (2)", "Elementary (3)", "Elementary (4)","Elementary (5)","Middle School (6)","Middle School (7)","Middle School (8)", "High School (9-12)", "College Prep"])
    subjects = st.multiselect("Subjects (select or type)", ["Math", "Science", "English", "History", "Other"], default=[])
    if subjects and "Other" in subjects:
        other_subject = st.text_input("Specify 'Other' subject:")
        if other_subject:
            subjects.remove("Other")
            subjects.append(other_subject)
    st.info(f"Profile: {grade} | Subjects: {', '.join(subjects)}")

# Initialize chat history
if "messages" not in st.session_state:
    # Initial system message based on profile
    system_prompt = f"""
    You are a friendly, patient tutor for a {grade.lower()} student in {', '.join(subjects)}.
    Keep explanations simple, engaging, and step-by-step. Use examples from everyday life.
    For lessons: Break into short sections with objectives.
    For quizzes: Create 3-5 multiple-choice or short-answer questions with answers explained.
    For explanations: Use analogies and visuals (describe them).
    Suggest free resources like Khan Academy videos when relevant.
    Always encourage the student and ask what they'd like next.
    """
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# Display chat messages
for message in st.session_state.messages[1:]:  # Skip system
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your subjects! E.g., 'Explain fractions' or 'Quiz me on algebra'"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking and preparing your lesson..."):
            # Include full history for context
            full_messages = st.session_state.messages
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use gpt-4o if available for better results
                messages=full_messages,
                max_tokens=500,
                temperature=0.7
            )
            llm_response = response.choices[0].message.content
            st.markdown(llm_response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": llm_response})

# Footer
st.markdown("---")
st.markdown("*Powered by OpenAI & Streamlit. For questions, contact support@example.com*")