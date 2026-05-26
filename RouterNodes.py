from typing import Literal

from langgraph.constants import END
from langgraph.types import Command
from packaging.metadata import parse_email
from pydantic import BaseModel, Field

from Models import get_llm
from Prompt import triage_system_prompt
from Schemas import State, RouterSchema


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
        goto = END

    elif result.classification == "reject":
        print("🚫 Classification: REJECT")
        goto = END
    elif result.classification == "refine":
        # If real life, this would do something else
        print("🔔 Classification: REFINE")
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto)