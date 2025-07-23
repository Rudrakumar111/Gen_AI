from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import streamlit as st
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

def get_openai_response(question):

    llm = ChatGroq(
        model="Gemma2-9b-It",
        api_key=groq_api_key,
    )
    
    response  = llm([HumanMessage(content=question)])
    return response


st.set_page_config(page_title="Q&A Demo")

st.header("Langchain Application")

input = st.text_input("Input: ",key="input")
response = get_openai_response(input)

submit = st.button("Ask the question")

if submit:
    st.subheader("The Response is")
    st.write(response.content)
