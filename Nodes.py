from langgraph.graph import MessagesState
from langgraph.types import interrupt, Command
from langchain_core.messages import ToolMessage
from Models import get_llm, get_llm_with_tools, get_llm_with_calendar_tools
from Prompt import get_refine_prompt, get_accept_prompt, get_update_prompt, check_date_prompt, \
    get_check_schedule_context_prompt, get_kg_extraction_prompt, get_planning_prompt, \
    get_topic_extraction_from_feedback_prompt, get_final_plan_extraction_prompt
from RouterNodes import FinalPlan
from Schemas import State, ArticleData, KGExtraction

from base import get_tools_by_name
import re

from function_tool import save_to_neo4j, get_smart_schedule_dates, get_covered_context_from_neo4j


def call_llm(state: MessagesState):
    print("DEBUG - Cosa riceve il bot:")
    for msg in state["messages"]:
        print(f"  {msg.type}: {msg.content}")
    llm = get_llm()
    risposta = llm.invoke(state["messages"])
    return {"messages": [risposta]}


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

    # --- MODIFICA QUI LA LOGICA DELLA PRINT ---
    # Controlliamo l'ultimo messaggio per capire se stiamo rientrando da un tool
    messaggi = state.get("messages", [])
    sta_tornando_da_ricerca = False
    if messaggi:
        ultimo_messaggio = messaggi[-1]
        # Se l'ultimo messaggio è di tipo ToolMessage, stiamo ciclando in background
        if getattr(ultimo_messaggio, "type", "") == "tool":
            sta_tornando_da_ricerca = True

    if not sta_tornando_da_ricerca:
        print(f"\n⚙️ Avvio stesura articolo su: '{topic_da_scrivere}'")
    # ------------------------------------------

    llm = get_llm_with_tools()
    accept_prompt = get_accept_prompt(topic_da_scrivere)

    # CORREZIONE 1: Filtriamo lo storico dei messaggi.
    # Evitiamo di passare al LLM le stesure degli articoli precedenti,
    # andando a ritroso e fermandoci appena troviamo la vecchia stesura.
    storico_pulito = []
    for msg in reversed(state.get("messages", [])):
        # Se incontriamo la chiamata al tool di stesura del post precedente, ci fermiamo
        if hasattr(msg, "tool_calls") and any(tc.get("name") == "write_an_article" for tc in msg.tool_calls):
            break
        # Se incontriamo il messaggio effettivo del tool, ci fermiamo
        if getattr(msg, "name", "") == "write_an_article":
            break
        storico_pulito.insert(0, msg)

    # Passiamo al LLM solo il prompt di sistema e lo storico "pulito"
    messages = [{"role": "system", "content": accept_prompt}] + storico_pulito
    response = llm.invoke(messages)

    return Command(
        update={
            "messages": [response],
            "current_topic": topic_da_scrivere,
            "data_proposta": data_assegnata
        },
        goto="tool_node"
    )


# Assicurati di importare i tuoi tool
# dalla tua mappa, ad esempio: get_tools_by_name = {"write_an_article": write_an_article}




def tool_node(state: State):
    result = []
    last_message = state["messages"][-1]

    testo_articolo = ""
    articolo_generato = None

    # Eseguiamo TUTTI i tool che l'LLM ha richiesto
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})

        tool = get_tools_by_name[tool_name]
        observation = tool.invoke(tool_args)

        tool_message = ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"],
            name=tool_name
        )
        result.append(tool_message)

        # 🎯 Se è il tool di scrittura, estraiamo i dati
        if tool_name == "write_an_article":
            titolo_estratto = tool_args.get("about", "Nuovo Articolo")
            autore_estratto = tool_args.get("author", "Agente AI")
            testo_articolo = str(observation)

            # (Ho rimosso le lunghe print qui per evitare che stampi due volte
            # dato che Main.py le stampa già durante l'interrupt)
            print(f"\n✅ Articolo '{titolo_estratto}' generato con successo! Passo alla revisione umana...")

            articolo_generato = ArticleData(
                title=titolo_estratto,
                text=testo_articolo,
                author=autore_estratto,
                date=state.get("data_proposta")
            )

    # --- DECIDIAMO DOVE ANDARE ---
    if articolo_generato is not None:
        return Command(
            update={
                "messages": result,
                "final_article": articolo_generato
            },
            goto="ask_feedback_node"
        )
    else:
        # MODIFICA LA STAMPA QUI:
        nomi_tools_usati = [tc["name"] for tc in last_message.tool_calls]
        print(f"\n🛠️ [DEBUG AGENTE] In background l'LLM ha appena usato: {', '.join(nomi_tools_usati)}")
        return Command(
            update={"messages": result},
            goto="accept_node"
        )


def ask_feedback_node(state: State):
    # Recuperiamo l'articolo appena generato dallo stato
    articolo = state.get("final_article")

    # Estraiamo il testo in modo sicuro
    if isinstance(articolo, dict):
        testo_articolo = articolo.get("text", "")
    else:
        testo_articolo = getattr(articolo, "text", "")

    # Mettiamo in pausa il grafo.
    # Main.py intercetterà "articolo_generato" e si occuperà di stamparlo a schermo in modo pulito.
    feedback_utente = interrupt({"articolo_generato": testo_articolo})

    # Una volta ripreso, confezioniamo il feedback
    messaggio_feedback = HumanMessage(
        content=f"Questo è il feedback dell'utente sull'articolo appena scritto: {feedback_utente}"
    )

    # Andiamo al router che deciderà se approvare o riscrivere
    return Command(
        update={"messages": [messaggio_feedback]},
        goto="tool_node_router"
    )

def update_article_node(state: MessagesState):
    llm = get_llm_with_tools()
    update_prompt = get_update_prompt()

    # Costruiamo i messaggi da passare al LLM: il system prompt per la modifica + tutta la history
    messages = [{"role": "system", "content": update_prompt}] + state["messages"]

    response = llm.invoke(messages)

    # Passiamo il comando a tool_node per eseguire la nuova stesura
    return Command(update={"messages": [response]}, goto="tool_node")


def check_schedule_node(state: State):
    last_message = state["messages"][-1]
    llm = get_llm_with_calendar_tools()

    data_estratta = state.get("data_proposta")
    data_testo = data_estratta if data_estratta else "Nessuna data attualmente assegnata"
    n_days = state.get("n_days", 3)

    context_prompt = get_check_schedule_context_prompt(check_date_prompt, data_testo, n_days)

    ai_msg = llm.invoke([{"role": "system", "content": context_prompt}] + [last_message])
    new_messages = [ai_msg]

    if hasattr(ai_msg, "tool_calls") and len(ai_msg.tool_calls) > 0:
        print(f"🔧 L'LLM ha richiesto {len(ai_msg.tool_calls)} tool(s). Esecuzione in corso...")
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            selected_tool = get_tools_by_name[tool_name]
            tool_result = selected_tool.invoke(tool_args)

            tool_msg = ToolMessage(
                content=str(tool_result),
                name=tool_name,
                tool_call_id=tool_id
            )
            new_messages.append(tool_msg)

            match = re.search(r"\d{4}-\d{2}-\d{2}", str(tool_result))
            if match:
                data_estratta = match.group(0)
    else:
        print("✅ Nessun tool richiesto dall'LLM. Risposta generata direttamente.")

    print("\n" + "=" * 50)
    print("📅 RISULTATO SCHEDULING (Tool/LLM):")
    for msg in new_messages:
        msg.pretty_print()
    print("=" * 50 + "\n")

    # Passiamo al nuovo nodo di interazione invece che al router
    return Command(
        update={
            "messages": new_messages,
            "data_proposta": data_estratta
        },
        goto="ask_user_schedule_node"  # <-- NUOVO NODO
    )


def ask_user_schedule_node(state: State):
    # Recuperiamo i messaggi per capire cosa ha detto l'AI
    messaggi = state.get("messages", [])

    # Cerchiamo l'ultimo AIMessage per estrarne il testo
    # (Potrebbe esserci un ToolMessage alla fine, quindi dobbiamo assicurarci di avere il testo giusto)
    testo_assistente = "Ho elaborato le date. Come procediamo?"
    for msg in reversed(messaggi):
        if hasattr(msg, "content") and msg.content and getattr(msg, "type", "") == "ai":
            testo_assistente = msg.content
            break

    # Lanciamo l'interrupt
    user_feedback = interrupt({"schedule_result": testo_assistente})

    print(f"👤 Utente ha risposto: {user_feedback}")

    # Aggiorniamo lo stato con la risposta e andiamo al router decisionale
    return Command(
        update={
            "messages": [HumanMessage(content=user_feedback)]
        },
        goto="scheduling_node_router"
    )



from Models import get_llm  # Assicurati che sia importato
from Schemas import KGExtraction  # Assicurati che sia importato


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

    # 1. ESTRAZIONE: Ora prendiamo e rimuoviamo l'articolo dalla coda (pop)
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

    # 1.5 RECUPERO DEI TOPIC ESISTENTI DAL GRAFO (Memoria Semantica)
    from test_neo4j import graph  # Assicurati di importare l'istanza del grafo
    existing_topics = []
    if graph:
        try:
            records = graph.query("MATCH (t:Topic) RETURN t.name AS name")
            existing_topics = [r["name"] for r in records]
        except Exception as e:
            print(f"⚠️ Impossibile recuperare i topic esistenti: {e}")

    # 2. SALVATAGGIO SUL KNOWLEDGE GRAPH CON LOGICA AI-DRIVEN
    print("🧩 Estrazione Entità per il Knowledge Graph in corso...")
    llm = get_llm().with_structured_output(KGExtraction)

    # Prompt arricchito con i topic esistenti
    prompt_estrazione = get_kg_extraction_prompt(titolo, testo, existing_topics)

    estrazione = llm.invoke([{"role": "user", "content": prompt_estrazione}])

    # Passiamo anche i related_topics estratti dall'AI alla funzione di salvataggio
    save_to_neo4j(
        title=titolo,
        topic=estrazione.topic,
        claims=estrazione.claims,
        sources=estrazione.sources,
        publish_date=target_date,
        related_topics=estrazione.related_topics  # <--- Nuovo parametro
    )

    # 3. CONTROLLO CODA E AGGIORNAMENTO STATO
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
            update={"approved_articles": []},  # Svuotiamo definitivamente
            goto=END
        )

import datetime
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage

from Schemas import TopicSelection


def planning_node(state: State) -> Command:
    print("\n--- [planning_node] Generazione Piano Editoriale basato su KG ---")

    last_input = state["messages"][-1].content
    n = state.get("n_days", 3)

    # 1. Calcoliamo in modo intelligente le prossime 3 date disponibili
    date_sicure = get_smart_schedule_dates(n_days=n, total_posts=3)
    data_1, data_2, data_3 = date_sicure[0], date_sicure[1], date_sicure[2]

    contesto_kg_lista = get_covered_context_from_neo4j()
    contesto_kg_str = "\n".join(contesto_kg_lista)

    llm = get_llm()
    prompt_planning = get_planning_prompt(last_input, data_1, data_2, data_3, contesto_kg_str)

    response = llm.invoke([{"role": "system", "content": prompt_planning}])
    piano_generato = response.content

    feedback_utente = interrupt({
        "proposta_piano": piano_generato,
        "schedule_result": f"Il sistema ha pianificato i post con cadenza ogni {n} giorni. Approvi?"
    })

    # --- NUOVA LOGICA: ESTRAZIONE DEI TOPIC SCELTI DAL FEEDBACK ---
    print("🧠 Estrazione dei titoli selezionati in base al feedback...")
    extractor = get_llm().with_structured_output(TopicSelection)
    estrazione_prompt = get_topic_extraction_from_feedback_prompt(piano_generato, feedback_utente)
    scelta = extractor.invoke([{"role": "user", "content": estrazione_prompt}])

    pending = scelta.selected_topics
    if not pending:  # Fallback di sicurezza se non capisce il feedback
        pending = [last_input]

        # TRASFORMIAMO GLI OGGETTI IN DIZIONARI SICURI PER IL SALVATAGGIO
        pending_dicts = [{"title": p.title, "date": p.date} for p in pending]
    print(f"📌 Articoli messi in coda di scrittura: {pending}")

    data_oggi = datetime.date.today().strftime("%Y-%m-%d")

    return Command(
        update={
            "messages": [HumanMessage(content=str(feedback_utente))],
            "editorial_plan": piano_generato,
            "justification": piano_generato,
            "data_proposta": data_oggi,
            "pending_topics": pending  # <--- Carichiamo la coda!
        },
        goto="accept_node"
    )


def process_plan_node(state: State):
    print("\n🧠 Elaborazione della tua selezione...")
    user_feedback = state["messages"][-1].content
    original_plan = state.get("editorial_plan", "")

    llm = get_llm().with_structured_output(FinalPlan)
    prompt = get_final_plan_extraction_prompt(original_plan, user_feedback)
    risultato = llm.invoke([{"role": "user", "content": prompt}])

    return Command(
        update={"pending_posts": risultato.posts_to_write},
        goto="drafting_router"
    )

def save_draft_node(state: State):
    final_article = state.get("final_article")
    approved_articles = state.get("approved_articles", [])

    nuova_lista = list(approved_articles)
    nuova_lista.append(final_article)

    # RIMUOVIAMO L'ARTICOLO DALLA CODA SOLO DOPO CHE L'UTENTE HA APPROVATO
    pending = state.get("pending_topics", [])
    nuovi_pending = pending[1:] if pending else []

    return Command(
        update={
            "approved_articles": nuova_lista,
            "pending_topics": nuovi_pending   # <--- Ora si passa al successivo in modo sicuro
        },
        goto="drafting_router"
    )