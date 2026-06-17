from langgraph.graph import MessagesState
from langgraph.types import interrupt, Command
from langchain_core.messages import ToolMessage
from Models import get_llm, get_llm_with_tools, get_llm_with_calendar_tools
from Prompt import get_refine_prompt, get_accept_prompt, get_update_prompt, check_date_prompt
from RouterNodes import FinalPlan
from Schemas import State, ArticleData, KGExtraction
from Tools import save_to_neo4j, get_covered_context_from_neo4j, get_smart_schedule_dates
from base import get_tools_by_name
import re


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
    print(response.content)

    risposta_utente = interrupt({"proposta": response.content})
    return Command(
        update={"messages": [HumanMessage(content=risposta_utente)]},
        goto="triage_router"
    )


def accept_node(state: State):
    pending = state.get("pending_topics", [])

    # Leggiamo semplicemente il primo elemento in coda senza rimuoverlo
    if pending:
        elemento = pending[0]
        # Ora sappiamo per certo che è un dizionario
        if isinstance(elemento, dict):
            topic_da_scrivere = elemento.get("title", "Argomento generico")
            data_assegnata = elemento.get("date")
        else:
            topic_da_scrivere = getattr(elemento, "title", "Argomento generico")
            data_assegnata = getattr(elemento, "date", None)
    else:
        topic_da_scrivere = state.get("current_topic", "Argomento generico")
        data_assegnata = state.get("data_proposta", None)

    print(f"\n⚙️ Avvio/Ripresa stesura articolo su: '{topic_da_scrivere}'")

    llm = get_llm_with_tools()
    accept_prompt = get_accept_prompt(topic_da_scrivere)
    messages = [{"role": "system", "content": accept_prompt}] + state.get("messages", [])
    response = llm.invoke(messages)

    return Command(
        update={
            "messages": [response],
            "current_topic": topic_da_scrivere,
            "data_proposta": data_assegnata
            # NON ELIMINIAMO NIENTE DA PENDING_TOPICS QUI!
        },
        goto="tool_node"
    )


# Assicurati di importare i tuoi tool
# dalla tua mappa, ad esempio: get_tools_by_name = {"write_an_article": write_an_article}

def tool_node(state: State):
    result = []
    last_message = state["messages"][-1]

    # Variabili di appoggio
    testo_articolo = ""
    articolo_generato = None  # Lo usiamo come "bandierina" per capire se ha scritto l'articolo

    # Eseguiamo TUTTI i tool che l'LLM ha richiesto in questo turno
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})

        # Eseguiamo il tool
        tool = get_tools_by_name[tool_name]
        observation = tool.invoke(tool_args)

        # Creiamo il messaggio di risposta del tool per l'LLM
        tool_message = ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"],
            name=tool_name
        )
        result.append(tool_message)

        # 🎯 CONTROLLO CRITICO: È il tool di scrittura?
        # Sostituisci "write_an_article" con il VERO NOME del tuo tool di scrittura se diverso
        if tool_name == "write_an_article":
            titolo_estratto = tool_args.get("about", "Nuovo Articolo")
            autore_estratto = tool_args.get("author", "Agente AI")
            testo_articolo = str(observation)

            print("\n" + "=" * 50)
            print(f"📝 ARTICOLO GENERATO (Titolo: {titolo_estratto} | Autore: {autore_estratto}):")
            print("=" * 50)
            print(testo_articolo)
            print("=" * 50 + "\n")

            # Valorizziamo l'oggetto finale
            articolo_generato = ArticleData(
                title=titolo_estratto,
                text=testo_articolo,
                author=autore_estratto
            )

    # --- FUORI DAL CICLO FOR: DECIDIAMO DOVE ANDARE ---

    # CASO A: L'LLM ha usato il tool per scrivere l'articolo
    if articolo_generato is not None:
        # Chiediamo il feedback all'utente
        feedback_utente = interrupt({"articolo_generato": testo_articolo})

        messaggio_feedback = HumanMessage(
            content=f"Questo è il feedback dell'utente sull'articolo appena scritto: {feedback_utente}"
        )
        result.append(messaggio_feedback)

        # Aggiorniamo lo stato e andiamo alla fase di router per eventuale riscrittura
        return Command(
            update={
                "messages": result,
                "final_article": articolo_generato
            },
            goto="tool_node_router"
        )

    # CASO B: L'LLM ha fatto solo ricerche (Tavily, Neo4j, ecc.)
    else:
        print(f"\n🔍 L'agente ha consultato {len(last_message.tool_calls)} fonte/i in background. Torno a elaborare...")
        # Rimandiamo la palla ad accept_node in modo che legga i risultati delle ricerche e scriva l'articolo
        return Command(
            update={"messages": result},
            goto="accept_node"
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

    # 1. Recupera la data pre-calcolata dallo State
    data_estratta = state.get("data_proposta")
    data_testo = data_estratta if data_estratta else "Nessuna data attualmente assegnata"

    # 2. Arricchisci il System Prompt dinamicamente
    context_prompt = (
        f"{check_date_prompt}\n\n"
        f"--- INFORMAZIONE DI CONTESTO INTERNA ---\n"
        f"La data attualmente pianificata/proposta per questo articolo dal piano editoriale è: {data_testo}. "
        f"Se l'utente ti chiede quale data avevi pianificato o qual è la data proposta, rispondi comunicando questa esatta data."
    )

    # 3. Invoca l'LLM con il prompt arricchito
    ai_msg = llm.invoke([{"role": "system", "content": context_prompt}] + [last_message])
    new_messages = [ai_msg]

    # 3. Verifichiamo se l'LLM ha deciso di chiamare uno o più tool
    if hasattr(ai_msg, "tool_calls") and len(ai_msg.tool_calls) > 0:
        print(f"🔧 L'LLM ha richiesto {len(ai_msg.tool_calls)} tool(s). Esecuzione in corso...")

        # 4. Eseguiamo ogni tool richiesto
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"   -> Eseguo '{tool_name}' con argomenti: {tool_args}")

            # Richiamiamo la funzione Python vera e propria
            selected_tool = get_tools_by_name[tool_name]
            tool_result = selected_tool.invoke(tool_args)

            # Creiamo il ToolMessage con il risultato da dare in pasto all'LLM
            tool_msg = ToolMessage(
                content=str(tool_result),
                name=tool_name,
                tool_call_id=tool_id
            )
            new_messages.append(tool_msg)
            # 🎯 ESTRAZIONE DATA: Cerca il formato YYYY-MM-DD nel testo restituito dal tool
            match = re.search(r"\d{4}-\d{2}-\d{2}", str(tool_result))
            if match:
                data_estratta = match.group(0)

    else:
        print("✅ Nessun tool richiesto dall'LLM. Risposta generata direttamente.")

    # 6. Restituiamo tutti i nuovi messaggi generati (AIMessage(s) e ToolMessage(s))
    # LangGraph li appenderà automaticamente alla lista 'messages' dello State
        # --- NUOVA SEZIONE: STAMPA, INTERRUPT E AGGIORNAMENTO STATO ---

    # 1. Stampiamo il risultato (l'ultimo messaggio aggiunto, che sia il ToolMessage o l'AIMessage)
    print("\n" + "=" * 50)
    print("📅 RISULTATO SCHEDULING (Tool/LLM):")
    for msg in new_messages:
        # pretty_print() è un metodo comodo di LangChain per stampare i messaggi in modo leggibile
        msg.pretty_print()
    print("=" * 50 + "\n")

    # 2. Lanciamo l'interrupt.
    # Passiamo un dizionario in modo che Main.py possa riconoscerlo,
    # esattamente come hai fatto per "proposta" e "articolo_generato".
    #user_feedback = interrupt({"schedule_result": "In attesa di feedback sulle date..."})
    # Usa questo:
    testo_assistente = ai_msg.content if ai_msg.content else "Ho elaborato le date. Come procediamo?"
    user_feedback = interrupt({"schedule_result": testo_assistente})

    # 3. Aggiorniamo lo stato con l'input dell'utente
    print(f"👤 Utente ha risposto: {user_feedback}")
    new_messages.append(HumanMessage(content=user_feedback))

    return Command(
        update={
            "messages": new_messages,
            "data_proposta": data_estratta  # Usa il nome della variabile in cui hai salvato la data
        },
        goto="scheduling_node_router"
    )



from langgraph.constants import END

from Models import get_llm  # Assicurati che sia importato
from Schemas import KGExtraction  # Assicurati che sia importato


def decision_node(state: State) -> Command:
    print("--- [decision_node] Conferma Data e Schedulazione ---")

    target_date = state.get("data_proposta")
    if not target_date:
        target_date = "2026-01-01"

    final_article = state.get("final_article")

    if final_article:
        final_article.date = target_date
        # Estrazione sicura dict vs Pydantic
        if isinstance(final_article, dict):
            titolo = final_article.get("title", "Senza Titolo")
            testo = final_article.get("text", "")
            final_article["date"] = target_date
        else:
            titolo = final_article.title
            testo = final_article.text
            final_article.date = target_date
        print(f"✅ Articolo '{final_article.title}' confermato per la data {target_date}.")

        # 🧩 ESTRAZIONE E SALVATAGGIO SPOSTATI QUI
        print("🧩 Estrazione Entità per il Knowledge Graph in corso...")
        llm = get_llm().with_structured_output(KGExtraction)
        prompt_estrazione = f"Estrai il topic principale, massimo 3 affermazioni chiave (claims) e le fonti da questo testo.\nTitolo: {final_article.title}\nTesto: {final_article.text}"
        estrazione = llm.invoke([{"role": "user", "content": prompt_estrazione}])

        # Salviamo su Neo4j passando anche la data target!
        save_to_neo4j(final_article.title, estrazione.topic, estrazione.claims, estrazione.sources, target_date)

    else:
        print("⚠️ Errore: Nessun articolo finale trovato nello stato.")

    # 2. CONTROLLO CODA DI SCHEDULAZIONE
    approved_articles = state.get("approved_articles", [])

    if approved_articles:
        print(f"\n🔁 Ci sono ancora {len(approved_articles)} articoli in coda da schedulare. Passo al prossimo...")
        return Command(
            goto="scheduling_queue_router"
        )
    else:
        print("\n✅ Tutti gli articoli richiesti sono stati scritti, schedulati e salvati nel Knowledge Graph!")
        return Command(goto=END)


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

    # ... logica Knowledge Graph ...

    llm = get_llm()
    prompt_planning = (
        f"Sei l'Editor-in-Chief del blog. L'utente ha chiesto un articolo su: '{last_input}'.\n"
        # ... ometto parti esistenti per brevità ...
        f"DEVI ASSOLUTAMENTE ASSEGNARE QUESTE DATE ESATTE AI 3 POST, perché sono le uniche disponibili a sistema:\n"
        f"- Post 1: {data_1}\n"
        f"- Post 2: {data_2}\n"
        f"- Post 3: {data_3}\n\n"
        f"Rispondi formattando chiaramente:\n"
        f"PIANO EDITORIALE:\n- Post 1 ({data_1}): Titolo\n- Post 2 ({data_2}): Titolo\n- Post 3 ({data_3}): Titolo\n"
    )
    response = llm.invoke([{"role": "system", "content": prompt_planning}])
    piano_generato = response.content

    feedback_utente = interrupt({
        "proposta_piano": piano_generato,
        "schedule_result": f"Il sistema ha pianificato i post con cadenza ogni {n} giorni. Approvi?"
    })

    # --- NUOVA LOGICA: ESTRAZIONE DEI TOPIC SCELTI DAL FEEDBACK ---
    print("🧠 Estrazione dei titoli selezionati in base al feedback...")
    extractor = get_llm().with_structured_output(TopicSelection)
    estrazione_prompt = (
        f"Questo è il piano editoriale proposto:\n{piano_generato}\n\n"
        f"L'utente ha risposto così: '{feedback_utente}'.\n"
        f"Estrai solo ed esclusivamente i titoli completi degli articoli che l'utente ha scelto o approvato di scrivere."
    )
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
    prompt = f"Piano originale:\n{original_plan}\n\nFeedback utente:\n{user_feedback}\n\nEstrai SOLO gli argomenti che l'utente ha approvato."
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