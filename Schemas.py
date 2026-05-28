from langgraph.graph import add_messages, MessagesState
from pydantic import BaseModel, Field
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from typing import Literal, Optional


class ArticleData(BaseModel):
    title: str = Field(description="Il titolo dell'articolo")
    text: str = Field(description="Il contenuto completo dell'articolo")
    author: str = Field(default="AI Agent", description="L'autore dell'articolo")
    date: Optional[str] = Field(default=None, description="La data di pubblicazione schedulata (YYYY-MM-DD)")

class State(MessagesState):
    classification_decision: Literal["accept", "refine", "reject"]
    final_article: Optional[ArticleData] = None  # <--- Il nostro nuovo oggetto

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


class RouterSchemaScheduling(BaseModel):
    ragionamento: str = Field(
        description="Analizza se l'utente sta chiedendo informazioni sulle disponibilità (scheduling) o se ha scelto una data per pubblicare (decision)."
    )
    classification: Literal["scheduling", "decision"] = Field(
        description=(
            "- 'scheduling': Se l'utente chiede la prima data disponibile o vuole verificare una data specifica.\n"
            "- 'decision': Se l'utente conferma una data e vuole procedere con la pubblicazione."
        )
    )
    data_proposta: Optional[str] = Field(
        None,
        description="La data YYYY-MM-DD se l'utente ne ha menzionata una, altrimenti None."
    )