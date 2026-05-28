from typing import Optional, List, Dict
from langchain_core.tools import BaseTool
from Tools import write_an_article, find_first_available_date_tool, check_specific_date_tool


def get_tools(tool_names: Optional[List[str]] = None, include_gmail: bool = False) -> List[BaseTool]:
    all_tools = {
        "write_article": write_an_article,
        "find_first_available_date_tool": find_first_available_date_tool,
        "check_specific_date_tool": check_specific_date_tool
    }
    if tool_names is None:
        return list(all_tools.values())
    return [all_tools[name] for name in tool_names if name in all_tools]


get_tools_by_name = {
    "write_an_article": write_an_article,
    "find_first_available_date_tool": find_first_available_date_tool,
    "check_specific_date_tool": check_specific_date_tool
}