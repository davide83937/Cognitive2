from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# 1. Definisci lo stato del tuo grafo
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Definisci il tuo nodo principale (la logica)
def entry_node(state: State):
    # Qui metterai la tua logica di triage o l'agente
    return {"messages": ["Punto di accesso raggiunto!"]}

# 3. Costruisci il grafo
builder = StateGraph(State)

builder.add_node("primo_nodo", entry_node)

# 4. Definisci l'arco di ingresso
builder.add_edge(START, "primo_nodo")
builder.add_edge("primo_nodo", END)

# 5. Compila il grafo (punto di accesso vero e proprio)
#memory = MemorySaver()
app = builder.compile()#"""checkpointer=memory"""