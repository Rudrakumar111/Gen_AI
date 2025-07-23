import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
 
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
 
# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
 
# Initialize LangChain model with Groq
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
parser = StrOutputParser()
chain = model | parser
 
# Initialize FastAPI app
app = FastAPI()
 
# Define request model
class TranslationRequest(BaseModel):
    text: str
 
# Endpoint for translation
@app.post("/translate")
async def translate(req: TranslationRequest):
    messages = [
        SystemMessage(content="Translate the following from English to gujarati"),
        HumanMessage(content=req.text)
    ]
    result = chain.invoke(messages)
    return {"translation": result}