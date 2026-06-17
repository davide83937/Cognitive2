from langchain_core.messages import HumanMessage
from langgraph import graph

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command
from langgraph.graph import StateGraph, START
from Nodes import refine_node, accept_node, tool_node, update_article_node, check_schedule_node, decision_node, \
    planning_node, process_plan_node, save_draft_node
from RouterNodes import triage_router, tool_node_router, scheduling_node_router, drafting_router, \
    scheduling_queue_router
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
builder.add_node("tool_node", tool_node)
builder.add_node("tool_node_router", tool_node_router)
builder.add_node("update_article_node", update_article_node)
builder.add_node("scheduling_node_router", scheduling_node_router)
builder.add_node("check_schedule_node", check_schedule_node)
builder.add_node("decision_node", decision_node)
builder.add_node("planning_node", planning_node)
builder.add_node("process_plan_node", process_plan_node)
builder.add_node("drafting_router", drafting_router)
builder.add_node("save_draft_node", save_draft_node)
builder.add_node("scheduling_queue_router", scheduling_queue_router)

# 4. Definisci l'arco di ingresso
builder.add_edge(START, "triage_router")

# 5. Compila il grafo (punto di accesso vero e proprio)
serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ('Schemas', 'ArticleData'),
        ('Schemas', 'PlannedArticle')  # <--- AGGIUNGI QUESTA RIGA
    ]
)

# Inizializza il MemorySaver passandogli il serializzatore personalizzato
memory = MemorySaver(serde=serde)
app = builder.compile(checkpointer=memory)#"""checkpointer=memory"""


print("Inviando la domanda al grafo locale...")
content = input()
config = {"configurable": {"thread_id": "1"}}
output = app.invoke(Command(update={"messages": [HumanMessage(content=content)],
                                        "classification_decision": None}), config=config)
print(output["messages"][-1].content)

if __name__ == "__main__":
    """print("Inviando la domanda al grafo locale...")
    content = input()
    config = {"configurable": {"thread_id": "1"}}
    output = app.invoke(Command(update={"messages": [HumanMessage(content=content)],
                                        "classification_decision": None}), config=config)
    print(output["messages"][-1].content)"""
    while True:

        # 2. Controlla lo stato DOPO l'esecuzione
        snapshot = app.get_state(config)

        # Se il grafo è fermo su un interrupt, gestisci il suggerimento e la ripresa
        if snapshot.tasks and snapshot.tasks[0].interrupts:
            dati = snapshot.tasks[0].interrupts[0].value
            new_input = ""
            # CASO 1: L'interrupt proviene dal refine_node
            if "proposta" in dati:
                print(f"🤖 Suggerimento: {dati['proposta']}")
                new_input = input("Inserisci il tuo topic raffinato: ")
                # --- NUOVO CASO: L'interrupt proviene da planning_node ---
            elif "proposta_piano" in dati:
                print("\n" + "=" * 50)
                print("📅 PROPOSTA CALENDARIO EDITORIALE (Ogni n giorni):")
                print("=" * 50)
                print(dati["proposta_piano"])
                print("=" * 50 + "\n")
                new_input = input("Scrivi 'ok' per approvare o inserisci modifiche: ")
            elif "articolo_generato" in dati:
                print("\n" + "=" * 50)
                print("📝 ARTICOLO GENERATO:")
                print("=" * 50)
                print(dati["articolo_generato"])
                print("=" * 50 + "\n")

                new_input = input("Inserisci il tuo feedback per migliorare l'articolo: ")
                # NUOVO CASO: L'interrupt proviene da check_schedule_node
            elif "schedule_result" in dati:
                print(f"🤖 Assistente: {dati['schedule_result']}")
                new_input = input("Inserisci la tua risposta o conferma la data: ")

            # Riprendi il grafo usando la 'resume' dell'interrupt
            # Questo passa 'nuovo_topic' direttamente alla variabile che aspettava l'interrupt
            app.invoke(Command(resume=new_input), config=config)

