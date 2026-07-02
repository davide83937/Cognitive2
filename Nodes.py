from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from Models import get_llm, get_llm_with_tools, get_llm_with_calendar_tools
from Prompt import get_refine_prompt, get_accept_prompt, get_update_prompt, check_date_prompt, \
    get_check_schedule_context_prompt, get_kg_extraction_prompt, get_planning_prompt, \
    get_topic_extraction_from_feedback_prompt
from Schemas import KGExtraction, State, ArticleData, TopicSelection, EditorialPlanOutput
from base import get_tools_by_name
import re
from function_tool import save_to_neo4j, get_smart_schedule_dates, get_covered_context_from_neo4j
import datetime
from langgraph.types import interrupt, Command



def call_llm(state: MessagesState):
    print("DEBUG - Cosa riceve il bot:")
    for msg in state["messages"]:
        print(f"  {msg.type}: {msg.content}")
    llm = get_llm()
    risposta = llm.invoke(state["messages"])
    return {"messages": [risposta]}


def planning_node(state: State) -> Command:
    print("\n--- [planning_node] Generazione Piano Editoriale basato su KG ---")

    last_input = state["messages"][-1].content

    date_sicure = get_smart_schedule_dates(total_posts=3)
    data_1, data_2, data_3 = date_sicure[0], date_sicure[1], date_sicure[2]

    contesto_kg_lista = get_covered_context_from_neo4j()
    contesto_kg_str = "\n".join(contesto_kg_lista)

    prompt_planning = get_planning_prompt(last_input, data_1, data_2, data_3, contesto_kg_str)
    llm = get_llm().with_structured_output(EditorialPlanOutput)
    response = llm.invoke([{"role": "system", "content": prompt_planning}])

    return Command(
        update={
            "editorial_plan": response.plan,
            "justification": response.justification,
        },
        goto="ask_plan_feedback_node"
    )


def ask_plan_feedback_node(state: State) -> Command:
    print("\n--- [ask_plan_feedback_node] In attesa di approvazione ---")
    piano_generato = state.get("editorial_plan", "")
    feedback_utente = interrupt({
        "proposta_piano": piano_generato,
        "schedule_result": "Il sistema ha pianificato i post. Approvi la programmazione o vuoi suggerire modifiche?"
    })
    return Command(
        update={
            "messages": [HumanMessage(content=str(feedback_utente))]
        },
        goto="process_plan_node"
    )


def process_plan_node(state: State) -> Command:
    print("\n🧠 --- [process_plan_node] Elaborazione del feedback e accodamento ---")

    user_feedback = state["messages"][-1].content
    original_plan = state.get("editorial_plan", "")

    last_input = "Argomento generico"
    if len(state["messages"]) > 1:
        last_input = state["messages"][-2].content

    extractor = get_llm().with_structured_output(TopicSelection)
    estrazione_prompt = get_topic_extraction_from_feedback_prompt(original_plan, user_feedback)
    scelta = extractor.invoke([{"role": "user", "content": estrazione_prompt}])

    pending = scelta.selected_topics
    if not pending:
        pending = [last_input]

    print(f"📌 Articoli messi in coda di scrittura: {pending}")

    data_oggi = datetime.date.today().strftime("%Y-%m-%d")

    return Command(
        update={
            "data_proposta": data_oggi,
            "pending_topics": pending
        },
        goto="accept_node"
    )


def refine_node(state: MessagesState):
    last_input = state["messages"][-1].content
    refinement_prompt = get_refine_prompt(last_input.__str__())
    llm = get_llm()
    response = llm.invoke([{"role": "system", "content": refinement_prompt}])
    #print(response.content)

    risposta_utente = interrupt({"proposta": response.content})
    return Command(
        update={"messages": [HumanMessage(content=risposta_utente)]},
        goto="triage_router"
    )


def accept_node(state: State):
    pending = state.get("pending_topics", [])
    if pending:
        elemento = pending[0]
        if isinstance(elemento, dict):
            topic_da_scrivere = elemento.get("title", "Argomento generico")
            data_assegnata = elemento.get("date")
        else:
            topic_da_scrivere = getattr(elemento, "title", "Argomento generico")
            data_assegnata = getattr(elemento, "date", None)
    else:
        topic_da_scrivere = state.get("current_topic", "Argomento generico")
        data_assegnata = state.get("data_proposta", None)
    messaggi = state.get("messages", [])
    sta_tornando_da_ricerca = False
    if messaggi:
        ultimo_messaggio = messaggi[-1]
        if getattr(ultimo_messaggio, "type", "") == "tool":
            sta_tornando_da_ricerca = True
    if not sta_tornando_da_ricerca:
        print(f"\n⚙️ Avvio stesura articolo su: '{topic_da_scrivere}'")
    llm = get_llm_with_tools()
    # 2. Recuperi i kg_summaries dallo stato
    sommari = state.get("kg_summaries", "")

    # 3. Passi ENTRAMBI alla funzione!
    accept_prompt = get_accept_prompt(topic_da_scrivere, sommari)
    storico_pulito = []
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "tool_calls") and any(tc.get("name") == "write_an_article" for tc in msg.tool_calls):
            break
        if getattr(msg, "name", "") == "write_an_article":
            break
        storico_pulito.insert(0, msg)
    messages = [{"role": "system", "content": accept_prompt}] + storico_pulito
    response = llm.invoke(messages)
    if response.content:
        print(f"\n💭 [THOUGHT AGENTE]: {response.content}")
    return Command(
        update={
            "messages": [response],
            "current_topic": topic_da_scrivere,
            "data_proposta": data_assegnata
        },
        goto="tool_node"
    )


def tool_node(state: State):
    result = []
    last_message = state["messages"][-1]
    articolo_generato = None

    # 1. Recupera il contatore dallo stato (default a 0)
    current_search_count = state.get("search_count", 0)
    state_updates = {}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})

        # ---> INIZIO GUARDRAIL LIMITE RICERCHE <---
        if tool_name == "verified_internet_search":
            current_search_count += 1

            if current_search_count > 2:  # Limite massimo di 2 ricerche
                msg_limite = "SYSTEM WARNING: Limite di 2 ricerche web raggiunto per questo post. Usa le informazioni già raccolte (fonti locali e web) per scrivere la bozza ORA con 'write_an_article', senza altre ricerche."
                print(f"🛑 [GUARDRAIL] {msg_limite}")

                # Restituiamo l'avviso come finta "Observation"
                result.append(ToolMessage(
                    content=msg_limite,
                    tool_call_id=tool_call["id"],
                    name=tool_name
                ))
                continue  # Salta l'esecuzione reale del tool e passa al prossimo (se c'è)
        # ---> FINE GUARDRAIL <---

        # Esecuzione normale degli altri tool o della ricerca se entro i limiti
        tool = get_tools_by_name[tool_name]
        observation = tool.invoke(tool_args)

        tool_message = ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"],
            name=tool_name
        )
        result.append(tool_message)

        # ---> INIZIO ESTRAZIONE KG SUMMARIES <---
        # Se il tool usato è quello del Knowledge Graph, salviamo il summary esplicitamente nello stato
        if tool_name in ["get_enhanced_topic_context"]:
            state_updates["kg_summaries"] = str(observation)
        # ---> FINE ESTRAZIONE KG SUMMARIES <---

        if tool_name == "write_an_article":
            titolo_estratto = tool_args.get("about", "Nuovo Articolo")
            autore_estratto = tool_args.get("author", "Agente AI")
            testo_articolo = str(observation)

            print(f"\n✅ Articolo '{titolo_estratto}' generato con successo! Passo alla revisione umana...")

            articolo_generato = ArticleData(
                title=titolo_estratto,
                text=testo_articolo,
                author=autore_estratto,
                date=state.get("data_proposta")
            )

    state_updates["messages"] = result
    if articolo_generato is not None:
        return Command(
            update={
                "messages": result,
                "final_article": articolo_generato,
                "search_count": 0  # <--- RESETTA IL CONTATORE per il prossimo articolo
            },
            goto="ask_feedback_node"
        )
    else:
        nomi_tools_usati = [tc["name"] for tc in last_message.tool_calls]
        print(f"\n🛠️ [DEBUG AGENTE] In background l'LLM ha appena usato: {', '.join(nomi_tools_usati)}")

        # AGGIUNTA GUARDRAIL
        if not last_message.tool_calls:
            print("⚠️ L'LLM non ha chiamato alcun tool nativo. Forza l'uso di write_an_article.")
            msg_forzatura = HumanMessage(
                content="SYSTEM WARNING: Non hai chiamato alcun tool. Usa esplicitamente il tool 'write_an_article' ora per generare il pezzo e proseguire.")
            result.append(msg_forzatura)

        return Command(
            update={
                "messages": result,
                "search_count": current_search_count
            },
            goto="accept_node"
        )


def ask_feedback_node(state: State):
    articolo = state.get("final_article")

    if isinstance(articolo, dict):
        testo_articolo = articolo.get("text", "")
    else:
        testo_articolo = getattr(articolo, "text", "")

    feedback_utente = interrupt({"articolo_generato": testo_articolo})

    messaggio_feedback = HumanMessage(
        content=f"Questo è il feedback dell'utente sull'articolo appena scritto: {feedback_utente}"
    )

    return Command(
        update={"messages": [messaggio_feedback]},
        goto="tool_node_router"
    )

def update_article_node(state: MessagesState):
    llm = get_llm_with_tools()
    update_prompt = get_update_prompt()
    messages = [{"role": "system", "content": update_prompt}] + state["messages"]
    response = llm.invoke(messages)
    return Command(update={"messages": [response]}, goto="tool_node")


def save_draft_node(state: State):
    final_article = state.get("final_article")
    approved_articles = state.get("approved_articles", [])

    nuova_lista = list(approved_articles)
    nuova_lista.append(final_article)

    pending = state.get("pending_topics", [])
    nuovi_pending = pending[1:] if pending else []

    return Command(
        update={
            "approved_articles": nuova_lista,
            "pending_topics": nuovi_pending   # <--- Ora si passa al successivo in modo sicuro
        },
        goto="drafting_router"
    )


def ask_schedule_node(state: State) -> Command:
    approved_articles = state.get("approved_articles", [])
    next_article = approved_articles[0]

    if isinstance(next_article, dict):
        titolo_articolo = next_article.get("title", "Articolo")
        data_precalcolata = next_article.get("date")
    else:
        titolo_articolo = next_article.title
        data_precalcolata = getattr(next_article, "date", None)

    if data_precalcolata:
        testo_guida = f"L'articolo '{titolo_articolo}' è attualmente pianificato per il {data_precalcolata}. Confermi questa data o preferisci verificarne altre?"
    else:
        testo_guida = f"Per quando vuoi schedulare l'articolo '{titolo_articolo}'?"

    risposta_utente = interrupt({"schedule_result": testo_guida})

    return Command(
        update={
            "data_proposta": data_precalcolata,
            "messages": [
                AIMessage(content=testo_guida),
                HumanMessage(content=risposta_utente)
            ]
        },
        goto="scheduling_node_router"
    )


def check_schedule_node(state: State):
    # Recuperiamo l'intera cronologia recente per dare pieno contesto all'LLM grande
    messages_history = state.get("messages", [])
    llm = get_llm_with_calendar_tools()

    data_estratta = state.get("data_proposta")
    data_testo = data_estratta if data_estratta else "Nessuna data attualmente assegnata"
    #n_days = state.get("n_days", 3)

    context_prompt = get_check_schedule_context_prompt(check_date_prompt, data_testo)

    # Prepariamo i messaggi per l'LLM inserendo il system prompt aggiornato
    input_messages = [{"role": "system", "content": context_prompt}] + messages_history

    # Passo 1: L'LLM decide se chiamare lo strumento di controllo del calendario
    ai_msg = llm.invoke(input_messages)
    new_messages = [ai_msg]

    if hasattr(ai_msg, "tool_calls") and len(ai_msg.tool_calls) > 0:
        print(f"🔧 L'LLM ha richiesto {len(ai_msg.tool_calls)} tool(s). Esecuzione in corso...")
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            selected_tool = get_tools_by_name[tool_name]
            tool_result = selected_tool.invoke(tool_args)

            tool_message = ToolMessage(
                content=str(tool_result),
                name=tool_name,
                tool_call_id=tool_id
            )
            new_messages.append(tool_message)

        # --- SOLUZIONE REACT ---
        # Passiamo l'output del tool di nuovo all'LLM grande (GPT-4)
        # In questo modo leggerà i dati grezzi del DB e applicherà la logica flessibile
        final_ai_msg = llm.invoke(input_messages + new_messages)
        new_messages.append(final_ai_msg)
    else:
        print("✅ Nessun tool richiesto dall'LLM. Risposta generata direttamente.")

    print("\n" + "=" * 50)
    print("📅 RISULTATO SCHEDULING (Elaborato da LLM Grande):")
    for msg in new_messages:
        msg.pretty_print()
    print("=" * 50 + "\n")

    # SICUREZZA STATO (Punto 4): Non aggiorniamo data_proposta qui.
    # Lo stato cambierà solo nel router quando l'utente darà la conferma definitiva.
    return Command(
        update={
            "messages": new_messages
        },
        goto="ask_user_schedule_node"
    )


def ask_user_schedule_node(state: State):
    messaggi = state.get("messages", [])

    testo_assistente = "Ho elaborato le date. Come procediamo?"
    for msg in reversed(messaggi):
        if hasattr(msg, "content") and msg.content and getattr(msg, "type", "") == "ai":
            testo_assistente = msg.content
            break

    user_feedback = interrupt({"schedule_result": testo_assistente})

    print(f"👤 Utente ha risposto: {user_feedback}")

    return Command(
        update={
            "messages": [HumanMessage(content=user_feedback)]
        },
        goto="scheduling_node_router"
    )



def decision_node(state: State) -> Command:
    print("--- [decision_node] Conferma Data e Schedulazione ---")

    target_date = state.get("data_proposta")
    if not target_date:
        target_date = "2026-01-01"

    approved_articles = state.get("approved_articles", [])

    if not approved_articles:
        print("⚠️ Errore: Nessun articolo da schedulare trovato nella coda.")
        from langgraph.constants import END
        return Command(goto=END)

    current_article = approved_articles.pop(0)

    # Estrazione sicura dict vs Pydantic
    if isinstance(current_article, dict):
        titolo = current_article.get("title", "Senza Titolo")
        testo = current_article.get("text", "")
        current_article["date"] = target_date
    else:
        titolo = current_article.title
        testo = current_article.text
        current_article.date = target_date

    print(f"✅ Articolo '{titolo}' confermato per la data {target_date}.")

    from test_neo4j import graph
    existing_topics = []
    if graph:
        try:
            records = graph.query("MATCH (t:Topic) RETURN t.name AS name")
            existing_topics = [r["name"] for r in records]
        except Exception as e:
            print(f"⚠️ Impossibile recuperare i topic esistenti: {e}")

    print("🧩 Estrazione Entità per il Knowledge Graph in corso...")
    llm = get_llm().with_structured_output(KGExtraction)

    prompt_estrazione = get_kg_extraction_prompt(titolo, testo, existing_topics)

    estrazione = llm.invoke([{"role": "user", "content": prompt_estrazione}])

    save_to_neo4j(
        title=titolo,
        topic=estrazione.topic,
        claims=estrazione.claims,
        sources=estrazione.sources,
        publish_date=target_date,
        related_topics=estrazione.related_topics
    )

    if approved_articles:
        print(f"\n🔁 Ci sono ancora {len(approved_articles)} articoli in coda da schedulare. Passo al prossimo...")
        return Command(
            update={"approved_articles": approved_articles},  # Salviamo la coda ridotta
            goto="scheduling_queue_router"
        )
    else:
        print("\n✅ Tutti gli articoli richiesti sono stati scritti, schedulati e salvati nel Knowledge Graph!")
        from langgraph.constants import END
        return Command(
            update={"approved_articles": []},
            goto=END
        )



