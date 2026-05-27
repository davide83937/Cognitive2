from langgraph.graph import add_messages, MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Annotated, Literal


class State(MessagesState):
    classification_decision: Literal["accept", "refine", "reject"]

class RouterSchema(BaseModel):
    ragionamento: str = Field(
        description="Analizza il topic. Spiega se ha una base scientifica, se è troppo generico o se è completamente fuori tema."
    )
    classification: Literal["accept", "refine", "reject"] = Field(
        description=(
            "DECISION RULES:\n"
            "- 'accept': Solo se l'utente propone un argomento abbastanza preciso.\n"
            "- 'refine': Solo se l'argomento è vago e non è stato ancora raffinato.\n"
            "- 'reject': Solo se fuori tema."
        )
    )

class RouterSchemaToolNode(BaseModel):
    ragionamento: str = Field(
        description="Analizza il feedback utente e spiega come mai lo hai interpretato in un certo modo"
    )
    classification: Literal["approve", "refine"] = Field(
        description=(
            "DECISION RULES:\n"
            "- 'approve': Se l'utente approva l'articolo così come è.\n"
            "- 'refine': Se l'utente ha chiesto delle modifiche.\n"
        )
    )

class StateInput(TypedDict):
    # This is the input to the state
    topic_input: str