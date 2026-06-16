import time
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import END
from langgraph.graph import MessagesState
from langgraph.types import interrupt, Command
from langchain_core.messages import ToolMessage
from Models import get_llm, get_llm_with_tools, get_llm_with_calendar_tools
from Notion_Stuff import add_row_to_notion, controlla_disponibilita_data
from Prompt import get_refine_prompt, get_accept_prompt, get_update_prompt, check_date_prompt
from Schemas import State, ArticleData, KGExtraction
from Tools import save_to_neo4j
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

def accept_node(state: MessagesState):
    last_input = state["messages"][-1].content
    llm = get_llm_with_tools()
    accept_prompt = get_accept_prompt(last_input.__str__())
    response = llm.invoke([{"role": "system", "content": accept_prompt}])
    return Command(update={"messages": [response]}, goto="tool_node")



# Assicurati di importare i tuoi tool
# dalla tua mappa, ad esempio: get_tools_by_name = {"write_an_article": write_an_article}

def tool_node(state: State):  # <--- Usa State al posto di MessagesState
    result = []
    last_message = state["messages"][-1]

    # Prepariamo delle variabili con valori di default
    titolo_estratto = "Nuovo Articolo"
    autore_estratto = "Agente AI"
    testo_articolo = ""

    for tool_call in last_message.tool_calls:
        # 1. ESTRAIAMO I PARAMETRI DIRETTAMENTE DAGLI ARGOMENTI DEL TOOL
        argomenti = tool_call.get("args", {})

        # Recuperiamo "about" e "author" se esistono
        titolo_estratto = argomenti.get("about", titolo_estratto)
        autore_estratto = argomenti.get("author", autore_estratto)

        # Esegue fisicamente la funzione passando gli argomenti
        tool = get_tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])

        testo_articolo = str(observation)

        # --- INIZIO STAMPA DEL RISULTATO ---
        print("\n" + "=" * 50)
        print(f"📝 ARTICOLO GENERATO (Titolo: {titolo_estratto} | Autore: {autore_estratto}):")
        print("=" * 50)
        print(testo_articolo)
        print("=" * 50 + "\n")
        # --- FINE STAMPA DEL RISULTATO ---

        tool_message = ToolMessage(
            content=testo_articolo,
            tool_call_id=tool_call["id"],
            name=tool_call["name"]
        )
        result.append(tool_message)

    # Chiediamo il feedback all'utente
    feedback_utente = interrupt({"articolo_generato": testo_articolo})

    messaggio_feedback = HumanMessage(
        content=f"Questo è il feedback dell'utente sull'articolo appena scritto: {feedback_utente}"
    )
    result.append(messaggio_feedback)

    # 2. CREIAMO L'OGGETTO ARTICOLO CON I DATI ESTRATTI
    articolo_generato = ArticleData(
        title=titolo_estratto,
        text=testo_articolo,
        author=autore_estratto
    )

    # Restituendo "messages" e "final_article", aggiorniamo la cronologia e salviamo l'oggetto
    return Command(
        update={
            "messages": result,
            "final_article": articolo_generato  # <--- Salviamo l'oggetto nello stato!
        },
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

    ai_msg = llm.invoke([{"role": "system", "content": check_date_prompt}] + [last_message])
    new_messages = [ai_msg]

    data_estratta = state.get("data_proposta")

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
    user_feedback = interrupt({"schedule_result": "In attesa di feedback sulle date..."})

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


# --- Il Nodo Decisionale Definitivo ---
def decision_node(state: State) -> Command:
    print("--- [decision_node] Verifica disponibilità finale (Senza LLM) ---")

    # 1. Recuperiamo la data salvata dal router
    target_date = state.get("data_proposta")
    if not target_date:
        target_date = "2026-01-01"  # Data fallback di emergenza se manca

    # 2. Chiamiamo la tua funzione di controllo
    risultato_disponibilita = controlla_disponibilita_data(target_date)

    # Se il risultato esiste ed è true
    is_available = risultato_disponibilita and risultato_disponibilita.get("is_available", False)

    if is_available:
        print("🚀 PUBBLICAZIONE ARTICOLO SU NOTION IN CORSO...")
        time.sleep(2)

        # Recuperiamo l'articolo generato per avere titolo, autore e testo
        final_article = state.get("final_article")

        if final_article:
            # Estraiamo i dati dall'oggetto Pydantic ArticleData
            titolo = final_article.title
            testo = final_article.text
            autore = final_article.author

            # Eseguiamo la pubblicazione
            add_row_to_notion(titolo, target_date, autore, testo)

            # Aggiorniamo la data nello stato dell'articolo
            final_article.date = target_date

            # --- NOVITÀ: Estrazione e salvataggio nel Knowledge Graph ---
            print("🧩 Estrazione Entità per il Knowledge Graph in corso...")
            llm = get_llm().with_structured_output(KGExtraction)

            # Chiediamo al LLM di analizzare il testo finale e restituirci l'oggetto strutturato
            prompt_estrazione = f"Estrai il topic principale, massimo 3 affermazioni chiave (claims) e le fonti da questo testo.\nTitolo: {titolo}\nTesto: {testo}"
            estrazione = llm.invoke([{"role": "user", "content": prompt_estrazione}])

            # Salviamo tutto in Neo4j
            save_to_neo4j(titolo, estrazione.topic, estrazione.claims, estrazione.sources)
            # -------------------------------------------------------------
        else:
            print("⚠️ Errore: Nessun articolo finale trovato nello stato da pubblicare.")

        return Command(
            update={"final_article": final_article},
            goto=END
        )
    else:
        # La data è piena, dobbiamo chiedere all'utente una nuova data
        msg_testo = f"Purtroppo la data {target_date} è già piena. Inserisci una nuova data in formato YYYY-MM-DD."
        ai_msg = AIMessage(content=msg_testo)

        # L'interruzione che aspetta il tuo input da terminale
        user_feedback = interrupt({"schedule_result": msg_testo})

        human_msg = HumanMessage(content=user_feedback)

        # Ritorniamo al router aggiungendo i messaggi in modo che l'LLM rianalizzi la nuova data
        return Command(
            update={"messages": [ai_msg, human_msg]},
            goto="scheduling_node_router"
        )