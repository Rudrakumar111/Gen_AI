import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Set environment variables (optional, if used)
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question: {question}")
    ]
)

# Initialize Streamlit UI
st.title("Langchain Demo With LLAMA3")
input_text = st.text_input("What question do you have in mind?")

# Initialize Ollama LLM
llm = Ollama(model="llama3.2")  

output_parser = StrOutputParser()

# Create the LangChain pipeline
chain = prompt | llm | output_parser

# Handle user input
if input_text:
    st.write(chain.invoke(input_text))
    
