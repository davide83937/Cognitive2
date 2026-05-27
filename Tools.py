from langchain_core.tools import tool

@tool
def write_an_article(about: str, author: str, content: str):
    """Write an article about the topic given by user, author is IA"""
    return f"My article about: {about}. \n{content} \n Written by {author}"