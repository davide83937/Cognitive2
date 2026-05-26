from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
def get_llm():
    return llm

