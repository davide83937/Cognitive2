from typing import Literal

from langgraph.constants import END, START
from langgraph.types import Command
from packaging.metadata import parse_email
from pydantic import BaseModel, Field

from Models import get_llm
from Prompt import triage_system_prompt, tool_node_prompt, scheduling_node_prompt
from Schemas import State, RouterSchema, RouterSchemaToolNode, RouterSchemaScheduling


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
        #goto = "accept_node"
        goto = "planning_node"

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
        goto = "scheduling_node_router"
        return Command(update={"messages": ["Quale è la prima data disponibile?"]}, goto=goto)

    elif result.classification == "refine":
        # If real life, this would do something else
        print("🔔 Classification: REFINE")
        print("User has asked some edits")
        goto = "update_article_node"
        return Command(goto=goto)
    else:
        raise ValueError(f"Invalid classification: {result.classification}")



def scheduling_node_router(state: State) -> Command[Literal["__end__"]]:
    feedback_input = state["messages"][-1].content

    system_prompt = scheduling_node_prompt
    user_prompt = feedback_input

    llm = get_llm()
    llm = llm.with_structured_output(RouterSchemaScheduling, method="json_mode")
    # 🎯 LA SVOLTA: Concateniamo il System Prompt con TUTTI i messaggi della storia ()
    result = llm.invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    classification = result.classification

    if classification == "decision":
        print("📧 Classification: DECISION")
        print("User has approved")
        goto = "decision_node"

    elif classification == "scheduling":
        # If real life, this would do something else
        print("🔔 Classification: SCHEDULING")
        print("You are talking about the schedule")
        goto = "check_schedule_node"

    if result.data_proposta and result.data_proposta != "NESSUNA":
        print(f"🎯 Trasferisco la nuova data allo stato: {result.data_proposta}")
        return Command(update={"data_proposta": result.data_proposta}, goto=goto)
    # Passiamo sia l'aggiornamento della data che la destinazione
    return Command(

        goto=goto
    )

