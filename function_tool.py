from Prompt import tavily_prompt
from query_neo4j import get_latest_scheduled_date_query, get_save_post_to_neo4j_query, get_covered_context_query, \
    get_future_post_counts_query
from test_neo4j import graph
from datetime import datetime, timedelta
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_latest_scheduled_date_from_db():
    if not graph:
        return None
    query = get_latest_scheduled_date_query()
    try:
        res = graph.query(query)
        if res and res[0].get("latest_date"):
            return res[0]["latest_date"]
    except Exception as e:
        print(f"Errore query ultima data: {e}")
    return None


def get_next_fixed_publish_date(base_date_str=None):

    # 0 = Lunedì, 2 = Mercoledì, 4 = Venerdì, 6 = Domenica
    giorni_pubblicazione = [0, 2, 4, 6]

    if base_date_str:
        #conversione da stringa a date
        base_date = datetime.strptime(base_date_str, "%Y-%m-%d").date()
    else:
        base_date = datetime.now().date()
    next_date = base_date + timedelta(days=1)

    while next_date.weekday() not in giorni_pubblicazione:
        next_date += timedelta(days=1)

    return next_date



tavily_search_tool = TavilySearchResults(
    max_results=3,
    description=tavily_prompt
)


def save_to_neo4j_old(title: str, topic: str, claims: list, sources: list, publish_date: str, related_topics: list = None):
    """
    Salva il post, le entità, la data di pubblicazione e genera le relazioni semantiche arricchite tra Topic.
    """
    if not graph:
        print("⚠️ Errore: Database Neo4j non connesso, salto il salvataggio.")
        return

    if related_topics is None:
        related_topics = []

    # Conversione sicura da oggetti Pydantic a dizionari
    formatted_relations = []
    for rel in related_topics:
        if hasattr(rel, "model_dump"):
            formatted_relations.append(rel.model_dump())
        elif hasattr(rel, "dict"):
            formatted_relations.append(rel.dict())
        else:
            formatted_relations.append(rel) # Fallback se è già un dizionario

    query = get_save_post_to_neo4j_query()

    try:
        graph.query(query, params={
            "title": title,
            "topic": topic,
            "claims": claims,
            "sources": sources,
            "date": publish_date,
            "related_topics": formatted_relations  # Usiamo la lista convertita
        })
        print("🧠 Knowledge Graph aggiornato con successo con relazioni AI-driven arricchite!")
    except Exception as e:
        print(f"❌ Errore durante il salvataggio nel KG: {e}")


def save_to_neo4j(title: str, topic: str, claims: list, sources: list, publish_date: str, documentation_text: str, related_topics: list = None):
    """
    Salva il post, le entità, la data di pubblicazione, il testo dell'articolo (Documentation)
    e genera le relazioni semantiche arricchite tra Topic.
    """
    if not graph:
        print("⚠️ Errore: Database Neo4j non connesso, salto il salvataggio.")
        return

    if related_topics is None:
        related_topics = []

    # Conversione sicura da oggetti Pydantic a dizionari
    formatted_relations = []
    for rel in related_topics:
        if hasattr(rel, "model_dump"):
            formatted_relations.append(rel.model_dump())
        elif hasattr(rel, "dict"):
            formatted_relations.append(rel.dict())
        else:
            formatted_relations.append(rel) # Fallback se è già un dizionario

    query = get_save_post_to_neo4j_query()

    try:
        graph.query(query, params={
            "title": title,
            "topic": topic,
            "claims": claims,
            "sources": sources,
            "date": publish_date,
            "documentation_text": documentation_text,  # <-- Parametro aggiunto per il nodo Documentation
            "related_topics": formatted_relations  # Usiamo la lista convertita
        })
        print("🧠 Knowledge Graph aggiornato con successo con relazioni AI-driven arricchite e documentazione salvata!")
    except Exception as e:
        print(f"❌ Errore durante il salvataggio nel KG: {e}")


def get_covered_context_from_neo4j():

    query = get_covered_context_query()

    try:
        records = graph.query(query)

        contesto_kg = []
        for r in records:
            topic = r.get("topic_name")

            claims_associati = [claim for claim in r.get("claims", []) if claim]

            info = f"Macro-Topic: '{topic}' (Claims già trattati: {claims_associati})"
            contesto_kg.append(info)

        return contesto_kg if contesto_kg else ["Nessun contenuto precedente nel KG"]
    except Exception as e:
        print(f"⚠️ Errore estrazione dettagliata Cypher: {e}")
        return ["Nessun contenuto precedente nel KG"]


def get_smart_schedule_dates(total_posts: int = 3) -> list[str]:
    from test_neo4j import graph
    from datetime import datetime

    if not graph:
        print("⚠️ Graph non connesso, calcolo date fittizie sul palinsesto.")
        date_pianificate = []
        base = get_next_fixed_publish_date(datetime.now().strftime("%Y-%m-%d"))
        for _ in range(total_posts):
            date_pianificate.append(base.strftime("%Y-%m-%d"))
            base = get_next_fixed_publish_date(base.strftime("%Y-%m-%d"))
        return date_pianificate

    oggi = datetime.now().date()
    giorno_corrente = datetime(oggi.year, oggi.month, oggi.day)

    if giorno_corrente.weekday() not in [0, 2, 4, 6]:
        giorno_corrente = get_next_fixed_publish_date(giorno_corrente.strftime("%Y-%m-%d"))

    query = get_future_post_counts_query()
    try:
        risultati = graph.query(query, params={"oggi": oggi.strftime("%Y-%m-%d")})
        conteggio = {res["post_date"]: res["count"] for res in risultati if res.get("post_date")}
    except Exception as e:
        print(f"Errore query conteggio: {e}")
        conteggio = {}

    date_pianificate = []

    for _ in range(total_posts):
        while True:
            data_str = giorno_corrente.strftime("%Y-%m-%d")
            occupazione_attuale = conteggio.get(data_str, 0)
            if occupazione_attuale < 3:
                date_pianificate.append(data_str)
                conteggio[data_str] = occupazione_attuale + 1
                if conteggio[data_str] >= 3:
                    giorno_corrente = get_next_fixed_publish_date(data_str)
                break
            else:
                giorno_corrente = get_next_fixed_publish_date(data_str)
    return date_pianificate



def setup_vector_database(cartella_documenti="./knowledge_base", db_path="./chroma_db"):
    # Se la cartella non esiste, la crea
    if not os.path.exists(cartella_documenti):
        os.makedirs(cartella_documenti)
        print(f"📁 Cartella '{cartella_documenti}' creata. Inserisci qui i tuoi PDF.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(db_path):
        return Chroma(persist_directory=db_path, embedding_function=embeddings)

    print("📚 Costruzione del Vector Database in corso dai PDF...")
    documenti = []
    for file in os.listdir(cartella_documenti):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(cartella_documenti, file))
            documenti.extend(loader.load())

    if not documenti:
        print("⚠️ Nessun PDF trovato. Il RAG non avrà documenti.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documenti)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=db_path)
    return vectorstore

def getRetriever():
    vectorstore_db = setup_vector_database()
    retriever = vectorstore_db.as_retriever(search_kwargs={"k": 3}) if vectorstore_db else None
    return retriever
