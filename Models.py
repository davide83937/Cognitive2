from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from Tools import write_an_article

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
def get_llm():
    return llm

def get_llm_with_tools():
    llm = get_llm()
    llm = llm.bind_tools([write_an_article])
    return llm
