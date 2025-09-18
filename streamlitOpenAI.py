# Importing the necessary modules from the Streamlit and LangChain packages
import streamlit as st  
from openai import OpenAI
 
 
openai_api_key = st.secrets.get("OPENAI_API_KEY")

# Setting the title of the Streamlit application
st.title('Senthil LLM 🤖')

# Creating a sidebar input widget for the OpenAI API key, input type is password for security
#openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

print(openai_api_key)

# Defining a function to generate a response using the OpenAI model
def generate_response(input_text):
    print(input_text) 

    # Initializing the OpenAI model with a specified temperature and API key
    #llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

    client = OpenAI(
        api_key=openai_api_key
    )

    response = client.responses.create(
    model="gpt-5-nano",
    input=input_text,
    store=True,
    )

    print(response.output_text);
 
    st.info(response.output_text)
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
    
    # If the form is submitted and the API key is valid, generate a response
    if submitted:
        generate_response(text)