from typing import Literal

from langgraph.constants import END, START
from langgraph.types import Command
from packaging.metadata import parse_email
from pydantic import BaseModel, Field

from Models import get_llm
from Prompt import triage_system_prompt, tool_node_prompt
from Schemas import State, RouterSchema, RouterSchemaToolNode


class TriageDecision(BaseModel):
    classification: Literal["accept", "refine", "reject"] = Field(description="La decisione di classificazione")

def triage_router(state: State) -> Command[Literal["__end__"]]:
    topic_input = state["messages"][-1].content
    system_prompt = triage_system_prompt

    user_prompt = topic_input

    # Run the router LLM
    llm = get_llm()
    llm = llm.with_structured_output(RouterSchema)
    result = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    classification = result.classification

    if classification == "accept":
        print("📧 Classification: ACCEPT")
        goto = "accept_node"

    elif result.classification == "reject":
        print("🚫 Classification: REJECT")
        goto = END
    elif result.classification == "refine":
        # If real life, this would do something else
        print("🔔 Classification: REFINE")
        goto = "refine_node"
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto)

def tool_node_router(state: State) -> Command[Literal["__end__"]]:
    article_input = state["messages"][-2].content
    feedback_input = state["messages"][-1].content
    system_prompt = tool_node_prompt
    user_prompt = feedback_input

    llm = get_llm()
    llm = llm.with_structured_output(RouterSchemaToolNode, method="json_mode")
    result = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    classification = result.classification

    if classification == "approve":
        print("📧 Classification: ACCEPT")
        print("User has approved")
        goto = END

    elif result.classification == "refine":
        # If real life, this would do something else
        print("🔔 Classification: REFINE")
        print("User has asked some edits")
        goto = "update_article_node"
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto)




