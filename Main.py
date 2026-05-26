from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from Nodes import call_llm
from RouterNodes import triage_router
from Schemas import State

#load_dotenv()
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")
#api_key = os.getenv("GROQ_API_KEY")


# 3. Costruisci il grafo
builder = StateGraph(State)

#builder.add_node("call_llm", call_llm)
builder.add_node("triage_router", triage_router)

# 4. Definisci l'arco di ingresso
builder.add_edge(START, "triage_router")

# 5. Compila il grafo (punto di accesso vero e proprio)
memory = MemorySaver()
app = builder.compile(checkpointer=memory)#"""checkpointer=memory"""

while True:
    print("Inviando la domanda al grafo locale...")
    content = input()
    config = {"configurable": {"thread_id": "1"}}
    output = app.invoke(Command(update={"messages": [HumanMessage(content=content)],
                                        "classification_decision": None}), config=config)
    print(output["messages"][-1].content)