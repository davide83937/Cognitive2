from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState
from langgraph.types import interrupt, Command
from langchain_core.messages import ToolMessage
from Models import get_llm, get_llm_with_tools
from Prompt import get_refine_prompt, get_accept_prompt
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

def tool_node(state: MessagesState):
    result = []
    # L'ultimo messaggio generato dal LLM, che contiene la richiesta dei tool
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        # Recupera il tool dalla tua mappa
        tool = get_tools_by_name[tool_call["name"]]

        # Esegue fisicamente la funzione passando gli argomenti
        observation = tool.invoke(tool_call["args"])

        # --- INIZIO STAMPA DEL RISULTATO ---
        #print("\n" + "=" * 50)
        #print("📝 ARTICOLO GENERATO:")
        #print("=" * 50)
        #print(observation)  # Qui stampi il contenuto reale!
        #print("=" * 50 + "\n")
        # --- FINE STAMPA DEL RISULTATO ---

        # Crea il messaggio formattato in modo nativo per LangChain
        tool_message = ToolMessage(
            content=str(observation),  # Forza a stringa per sicurezza
            tool_call_id=tool_call["id"],  # ID fondamentale per ricollegare la risposta
            name=tool_call["name"]  # Nome del tool eseguito
        )

        result.append(tool_message)
        # 1. Blocchiamo il grafo per chiedere il feedback
        # Passiamo l'articolo nel payload dell'interrupt così Main.py può stamparlo
    feedback_utente = interrupt({"articolo_generato": str(observation)})
    # 2. Aggiungiamo il feedback dell'utente come un nuovo HumanMessage
    messaggio_feedback = HumanMessage(
        content=f"Questo è il feedback dell'utente sull'articolo appena scritto: {feedback_utente}"
    )
    result.append(messaggio_feedback)
    # Restituendo "messages", LangGraph appenderà questi ToolMessage alla cronologia
    return Command(
        update={"messages": result},
        goto="tool_node_router"  # O il nome che hai dato al nodo successivo
    )




