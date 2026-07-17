from typing import Optional, List
from langchain_core.tools import BaseTool
from Tools import write_an_article,  rag_document_retriever, verified_internet_search, \
    intelligent_topic_matcher, \
    get_enhanced_topic_context, get_flexible_schedule_dates


def get_tools(tool_names: Optional[List[str]] = None) -> List[BaseTool]:
    all_tools = {
        "write_article": write_an_article,
        #"find_first_available_date_tool": find_first_available_date_tool,
        #"check_specific_date_tool": check_specific_date_tool,
        #"schedule_manager_tool": schedule_manager_tool,
        #"get_previous_topics": get_previous_topics,
        #"get_topic_claims": get_topic_claims,
        #"tavily_search_results_json": tavily_search_tool, # ATTENZIONE: Il nome interno che LangChain assegna a questo tool è questo!
        "rag_document_retriever": rag_document_retriever,
        "verified_internet_search": verified_internet_search,
        "intelligent_topic_matcher": intelligent_topic_matcher,
        "get_enhanced_topic_context": get_enhanced_topic_context,
        "get_flexible_schedule_dates": get_flexible_schedule_dates
    }
    if tool_names is None:
        return list(all_tools.values())
    return [all_tools[name] for name in tool_names if name in all_tools]


get_tools_by_name = {
    "write_an_article": write_an_article,
    #"find_first_available_date_tool": find_first_available_date_tool,
    #"check_specific_date_tool": check_specific_date_tool,
    #"get_previous_topics": get_previous_topics,
    #"get_topic_claims": get_topic_claims,
    # "tavily_search_results_json": tavily_search_tool,
    "get_flexible_schedule_dates": get_flexible_schedule_dates,
    "rag_document_retriever": rag_document_retriever,
    "verified_internet_search": verified_internet_search,
    "intelligent_topic_matcher": intelligent_topic_matcher,
    "get_enhanced_topic_context": get_enhanced_topic_context

}