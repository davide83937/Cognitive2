from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import END, START
from langgraph.types import Command, interrupt

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

    # Arricchiamo il prompt per FORZARE l'estrazione della data dell'utente
    system_prompt = (
        f"{scheduling_node_prompt}\n\n"
        f"ATTENZIONE: Leggi attentamente l'ultimo messaggio dell'utente: '{feedback_input}'.\n"
        f"Se l'utente APPROVA o SPECIFICA chiaramente una data (es. 'approvo la data del 23 giugno 2026', 'sposta al 26'), "
        f"devi OBBLIGATORIAMENTE estrarla nel formato YYYY-MM-DD (es. '2026-06-23') e inserirla nel campo 'data_proposta'. "
        f"La volontà scritta dell'utente ha priorità assoluta su qualsiasi altra precedente elaborazione."
    )

#    user_prompt = feedback_input

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

    # Se l'LLM ha estratto la tua data (es. 2026-06-23), la sovrascrive annullando l'errore del tool
    if result.data_proposta and result.data_proposta != "NESSUNA":
        print(f"🎯 Trasferisco la nuova data confermata dall'utente allo stato: {result.data_proposta}")
        return Command(update={"data_proposta": result.data_proposta}, goto=goto)

    return Command(goto=goto)

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
        from langgraph.constants import END
        return Command(goto=END)

    # GUARDAMO il primo elemento SENZA rimuoverlo dalla coda
    next_article = approved_articles[0]

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

    # Mettiamo in pausa il grafo
    risposta_utente = interrupt({"schedule_result": testo_guida})

    return Command(
        update={
            # NON AGGIORNIAMO LA CODA QUI. Passiamo solo i messaggi e la data
            "data_proposta": data_precalcolata,
            "messages": [
                AIMessage(content=testo_guida),
                HumanMessage(content=risposta_utente)
            ]
        },
        goto="scheduling_node_router"
    )