from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from Tools import write_an_article

load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

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

def get_notion_token():
    return NOTION_TOKEN

def get_notion_db_id():
    return NOTION_DATABASE_ID