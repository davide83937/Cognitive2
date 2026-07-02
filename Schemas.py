from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional, List


class ArticleData(BaseModel):
    title: str = Field(description="Il titolo dell'articolo")
    text: str = Field(description="Il contenuto completo dell'articolo")
    author: str = Field(default="AI Agent", description="L'autore dell'articolo")
    date: Optional[str] = Field(default=None, description="La data di pubblicazione schedulata (YYYY-MM-DD)")

class State(MessagesState):
    classification_decision: Literal["accept", "refine", "reject"]
    final_article: Optional[ArticleData] = None
    data_proposta: Optional[str] = None
    search_count: int
    editorial_plan: Optional[list[dict]] = None
    kg_summaries: str
    justification: Optional[str] = None
    current_topic: Optional[str] = None
    pending_topics: list[str] = Field(default_factory=list)
    approved_articles: list[ArticleData] = Field(default_factory=list)

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
    data_proposta: Optional[str] = Field(
        default=None,
        description="SOLO SE l'utente scrive esplicitamente una data (es. 'pubblica il 28 maggio'), estraila in formato YYYY-MM-DD. Se l'utente dice solo 'ok', 'va bene' o 'confermo', lascia null."
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

# --- NUOVO SCHEMA PER LE RELAZIONI ---
class TopicRelationship(BaseModel):
    target_topic: str = Field(description="Il nome ESATTO del topic esistente a cui ci stiamo collegando")
    relationship_type: Literal["PREREQUISITO", "SOTTO_CATEGORIA", "ESTENSIONE", "CONTRASTO", "APPLICAZIONE", "SIMILARE"] = Field(
        description="Il tipo specifico di relazione semantica"
    )
    reason: str = Field(description="Breve e logica spiegazione del perché questi due topic sono collegati")

class KGExtraction(BaseModel):
    topic: str = Field(description="Il topic principale di questo specifico articolo")
    claims: List[str] = Field(description="Massimo 3 affermazioni chiave dell'articolo")
    sources: List[str] = Field(description="Fonti citate")
    related_topics: List[TopicRelationship] = Field( # <-- AGGIORNATO QUI
        default=[],
        description="Lista di relazioni dettagliate con i topic esistenti. Aggiungi una relazione SOLO se c'è un nesso logico forte e verificabile."
    )

class PlannedArticle(BaseModel):
    title: str = Field(description="Il titolo completo dell'articolo pianificato")
    date: str = Field(description="La data esatta assegnata nel piano editoriale (formato YYYY-MM-DD)")

class TopicSelection(BaseModel):
    selected_topics: list[PlannedArticle] = Field(description="Lista degli articoli approvati dall'utente, ognuno con la sua data")


# --- NUOVO SCHEMA PER LE RELAZIONI ---
class TopicRelationship(BaseModel):
    target_topic: str = Field(description="Il nome ESATTO del topic esistente a cui ci stiamo collegando")
    relationship_type: Literal["PREREQUISITO", "SOTTO_CATEGORIA", "ESTENSIONE", "CONTRASTO", "APPLICAZIONE", "SIMILARE"] = Field(
        description="Il tipo specifico di relazione semantica"
    )
    reason: str = Field(description="Breve e logica spiegazione del perché questi due topic sono collegati")

# ... (Lascia intatti gli altri schemi) ...

class EditorialArticle(BaseModel):
    # Questo ConfigDict forza Pydantic e OpenAI a rispettare lo schema Strict
    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="Titolo proposto per l'articolo")
    about: str = Field(description="Argomento o focus principale dell'articolo")


class EditorialPlanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Mantiene lo standard rigido richiesto da OpenAI
    plan: list[EditorialArticle] = Field(description="Lista degli articoli pianificati")
    justification: str = Field(description="Giustificazione delle scelte fatte, e cita alcuni topic già presenti che non hai voluto ripetere")