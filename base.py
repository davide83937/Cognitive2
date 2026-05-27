from typing import Optional, List, Dict
from langchain_core.tools import BaseTool
from Tools import write_an_article

def get_tools(tool_names: Optional[List[str]] = None, include_gmail: bool = False) -> List[BaseTool]:
    all_tools = {
        "write_article": write_an_article,
    }
    if tool_names is None:
        return list(all_tools.values())
    return [all_tools[name] for name in tool_names if name in all_tools]


def get_tools_by_name(tools: Optional[List[BaseTool]] = None) -> Dict[str, BaseTool]:
    if tools is None:
        tools = get_tools()
    return {tool.name: tool for tool in tools}