from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState
from langgraph.types import interrupt, Command
from langchain_core.messages import ToolMessage
from Models import get_llm, get_llm_with_tools, get_llm_with_calendar_tools
from Prompt import get_refine_prompt, get_accept_prompt, get_update_prompt
from Schemas import State, ArticleData
from base import get_tools_by_name


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



def schedule_node(state: MessagesState):
    llm = get_llm_with_calendar_tools()