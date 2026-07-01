from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os
from Tools import intelligent_topic_matcher, get_enhanced_topic_context, calendar_query_tool, \
    get_flexible_schedule_dates

load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

"""llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)"""
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
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

def get_llm_with_calendar_tools():
    llm = get_llm()
    llm = llm.bind_tools([get_flexible_schedule_dates])
    return llm


from Tools import write_an_article,  get_previous_topics,  rag_document_retriever, verified_internet_search


def get_llm_with_tools():
    llm = get_llm()
    # Aggiungiamo Tavily alla cintura degli attrezzi!
    llm = llm.bind_tools([
        write_an_article,
        get_previous_topics,
        get_enhanced_topic_context,
        verified_internet_search,
        rag_document_retriever,
        intelligent_topic_matcher
    ])
    return llm