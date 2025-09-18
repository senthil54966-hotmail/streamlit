# Importing the necessary modules from the Streamlit and LangChain packages
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langchain.llms import OpenAI

# Setting the title of the Streamlit application
st.title('Simple LLM-App 🤖')

# Creating a sidebar input widget for the OpenAI API key, input type is password for security
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

print(openai_api_key)

load_dotenv()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
MODEL = "microsoft/phi-4"

# chain = prompt | llm_model 
# result = chain.invoke(input="Dell")
# print(result)
llm = ChatOpenAI(model=MODEL, openai_api_base=OPENAI_API_BASE)



# Defining a function to generate a response using the OpenAI model
def generate_response(input_text):
    print(input_text) 

    # Initializing the OpenAI model with a specified temperature and API key
    #llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)




    messages = [
        ("system", "Answer questions"),
        ("user", "{text}")
    ]
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages(messages)

    # Create a chain to combine the prompt and LLM
    chain = prompt | llm

    # Invoke the chain with your input
    response = chain.invoke({"text": input_text})

    print(response.content) 
    st.info(response.content)
    #st.info(result)
    # Displaying the generated response as an informational message in the Streamlit app
    #st.info(llm(input_text))

# Creating a form in the Streamlit app for user input
with st.form('my_form'):
    # Adding a text area for user input with a default prompt
    text = st.text_area('Enter text:', '')
    print(text)

   

    # Adding a submit button for the form
    submitted = st.form_submit_button('Submit')
    # Displaying a warning if the entered API key does not start with 'sk-'
    if not openai_api_key.startswith('lm-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    # If the form is submitted and the API key is valid, generate a response
    if submitted and openai_api_key.startswith('lm-'):
        generate_response(text)