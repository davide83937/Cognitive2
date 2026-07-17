from typing import Literal
from langgraph.constants import END
from langgraph.types import Command
from pydantic import BaseModel, Field
from Models import get_llm
from Prompt import triage_system_prompt, tool_node_prompt, scheduling_node_prompt, get_scheduling_router_system_prompt
from Schemas import State, RouterSchema, RouterSchemaToolNode, RouterSchemaScheduling


"""class TriageDecision(BaseModel):
    classification: Literal["accept", "refine", "reject"] = Field(description="La decisione di classificazione")"""

def triage_router(state: State) -> Command[Literal["__end__"]]:
    topic_input = state["messages"][-1].content
    system_prompt = triage_system_prompt

   # user_prompt = topic_input

    llm = get_llm()
    llm = llm.with_structured_output(RouterSchema)
    result = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": topic_input},
        ]
    )
    classification = result.classification

    if classification == "accept":
        print("📧 Classification: ACCEPT")
        goto = "planning_node"

    elif result.classification == "reject":
        print("🚫 Classification: REJECT")
        goto = END
    elif result.classification == "refine":
        print("🔔 Classification: REFINE")
        goto = "refine_node"
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto)
"""
class FinalPlan(BaseModel):
    posts_to_write: list[str] = Field(description="Lista degli argomenti da scrivere, escludendo quelli scartati dall'utente")"""

def tool_node_router(state: State) -> Command[Literal["__end__"]]:
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
        print("📧 Classification: ACCEPT - User has approved the draft")
        return Command(goto="save_draft_node")

    elif result.classification == "refine":
        print("🔔 Classification: REFINE")
        print("User has asked some edits")
        goto = "update_article_node"
        return Command(goto=goto)
    else:
        raise ValueError(f"Invalid classification: {result.classification}")

def drafting_router(state: State) -> Command:
    pending_topics = state.get("pending_topics", [])

    if not pending_topics:
        print("\n✅ Tutti gli articoli scelti sono stati scritti e approvati! Passiamo alla schedulazione.")
        return Command(goto="scheduling_queue_router")

    print(f"\n⏳ Ci sono ancora {len(pending_topics)} articoli in coda. Riprendo la stesura...")

    return Command(goto="accept_node")


def scheduling_queue_router(state: State) -> Command:
    approved_articles = state.get("approved_articles", [])

    if not approved_articles:
        print("\n🏁 Tutte le schedulazioni completate! Il lavoro è finito.")
        from langgraph.constants import END
        return Command(goto=END)

    print(f"\n📅 Passiamo alla schedulazione del prossimo articolo in coda...")

    return Command(goto="ask_schedule_node")


def scheduling_node_router(state: State) -> Command[Literal["__end__"]]:
    feedback_input = state["messages"][-1].content

    system_prompt = get_scheduling_router_system_prompt(scheduling_node_prompt, feedback_input)

    llm = get_llm()
    llm = llm.with_structured_output(RouterSchemaScheduling, method="json_mode")
    result = llm.invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    classification = result.classification

    if classification == "decision":
        print("📧 Classification: DECISION")
        print("User has approved")
        goto = "decision_node"
    elif classification == "scheduling":
        print("🔔 Classification: SCHEDULING")
        print("You are talking about the schedule")
        goto = "check_schedule_node"

    if result.data_proposta and result.data_proposta != "NESSUNA":
        print(f"🎯 Trasferisco la nuova data confermata dall'utente allo stato: {result.data_proposta}")
        return Command(update={"data_proposta": result.data_proposta}, goto=goto)

    return Command(goto=goto)





