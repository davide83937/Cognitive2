from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.types import interrupt, Command

from Models import get_llm
from Prompt import get_refine_prompt, get_accept_prompt


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
    llm = get_llm()
    print(last_input)
    accept_prompt = get_accept_prompt(last_input.__str__())
    response = llm.invoke([{"role": "system", "content": accept_prompt}])
    print(response.content)




