from typing import Optional, List, Dict
from langchain_core.tools import BaseTool
from Tools import write_an_article, find_first_available_date_tool, check_specific_date_tool, \
    get_previous_topics, get_topic_claims, rag_document_retriever
from function_tool import tavily_search_tool



def get_tools(tool_names: Optional[List[str]] = None, include_gmail: bool = False) -> List[BaseTool]:
    all_tools = {
        "write_article": write_an_article,
        "find_first_available_date_tool": find_first_available_date_tool,
        "check_specific_date_tool": check_specific_date_tool,
        "get_previous_topics": get_previous_topics,
        "get_topic_claims": get_topic_claims,
        "tavily_search_results_json": tavily_search_tool, # ATTENZIONE: Il nome interno che LangChain assegna a questo tool è questo!
        "rag_document_retriever": rag_document_retriever
    }
    if tool_names is None:
        return list(all_tools.values())
    return [all_tools[name] for name in tool_names if name in all_tools]


get_tools_by_name = {
    "write_an_article": write_an_article,
    "find_first_available_date_tool": find_first_available_date_tool,
    "check_specific_date_tool": check_specific_date_tool,
"get_previous_topics": get_previous_topics,
        "get_topic_claims": get_topic_claims,
        "tavily_search_results_json": tavily_search_tool,
        "rag_document_retriever": rag_document_retriever

}