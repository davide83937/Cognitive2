from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage
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

class FinalPlan(BaseModel):
    posts_to_write: list[str] = Field(description="Lista degli argomenti da scrivere, escludendo quelli scartati dall'utente")

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
        print("📧 Classification: ACCEPT - User has approved the draft")
        return Command(goto="save_draft_node")  # <--- MODIFICA QUI

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


def drafting_router(state: State) -> Command:
    # 1. Usiamo la variabile corretta 'pending_topics'
    pending_topics = state.get("pending_topics", [])

    if not pending_topics:
        print("\n✅ Tutti gli articoli scelti sono stati scritti e approvati! Passiamo alla schedulazione.")
        return Command(goto="scheduling_queue_router")

    print(f"\n⏳ Ci sono ancora {len(pending_topics)} articoli in coda. Riprendo la stesura...")

    # 2. Non facciamo alcun '.pop()' qui perché accept_node lo fa già all'inizio del suo ciclo
    return Command(goto="accept_node")


def scheduling_queue_router(state: State) -> Command:
    approved_articles = state.get("approved_articles", [])

    if not approved_articles:
        print("\n🏁 Tutte le schedulazioni completate! Il lavoro è finito.")
        return Command(goto=END)

    next_article = approved_articles[0]
    rimanenti = approved_articles[1:]

    # Gestione sicura Oggetto/Dizionario
    if isinstance(next_article, dict):
        titolo_articolo = next_article.get("title", "Articolo")
        data_precalcolata = next_article.get("date")
    else:
        titolo_articolo = next_article.title
        data_precalcolata = getattr(next_article, "date", None)

    print(f"\n📅 Passiamo alla schedulazione di: {titolo_articolo}")

    if data_precalcolata:
        testo_guida = f"L'articolo '{titolo_articolo}' è attualmente pianificato per il {data_precalcolata}. Confermi questa data o preferisci verificarne altre?"
    else:
        testo_guida = f"Per quando vuoi schedulare l'articolo '{titolo_articolo}'?"

    # 🛑 INSERIAMO L'INTERRUPT QUI!
    # Mettiamo in pausa il grafo e mostriamo all'utente il testo_guida
    risposta_utente = interrupt({"schedule_result": testo_guida})

    return Command(
        update={
            "approved_articles": rimanenti,
            "final_article": next_article,
            "data_proposta": data_precalcolata,
            "messages": [
                AIMessage(content=testo_guida),         # Ora il testo_guida è correttamente registrato come messaggio dell'AI
                HumanMessage(content=risposta_utente)   # Salviamo la tua risposta ("confermo", "cambia data", ecc.)
            ]
        },
        goto="scheduling_node_router"
    )