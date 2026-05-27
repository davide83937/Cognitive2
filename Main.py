from langchain_core.messages import HumanMessage
from langgraph import graph

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langgraph.graph import StateGraph, START
from Nodes import refine_node, accept_node
from RouterNodes import triage_router
from Schemas import State

#load_dotenv()
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")
#api_key = os.getenv("GROQ_API_KEY")


# 3. Costruisci il grafo
builder = StateGraph(State)

#builder.add_node("call_llm", call_llm)
builder.add_node("triage_router", triage_router)
builder.add_node("refine_node", refine_node)
builder.add_node("accept_node", accept_node)

# 4. Definisci l'arco di ingresso
builder.add_edge(START, "triage_router")

# 5. Compila il grafo (punto di accesso vero e proprio)
memory = MemorySaver()
app = builder.compile(checkpointer=memory)#"""checkpointer=memory"""
print("Inviando la domanda al grafo locale...")
content = input()
config = {"configurable": {"thread_id": "1"}}
output = app.invoke(Command(update={"messages": [HumanMessage(content=content)],
                                        "classification_decision": None}), config=config)
print(output["messages"][-1].content)

while True:

    # 2. Controlla lo stato DOPO l'esecuzione
    snapshot = app.get_state(config)

    # Se il grafo è fermo su un interrupt, gestisci il suggerimento e la ripresa
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        # Recupera il suggerimento "chirurgico"
        dati = snapshot.tasks[0].interrupts[0].value
        print(f"🤖 Suggerimento: {dati['proposta']}")

        # Chiedi input all'utente
        nuovo_topic = input("Inserisci il tuo topic raffinato: ")

        # Riprendi il grafo usando la 'resume' dell'interrupt
        # Questo passa 'nuovo_topic' direttamente alla variabile che aspettava l'interrupt
        app.invoke(Command(resume=nuovo_topic), config=config)

