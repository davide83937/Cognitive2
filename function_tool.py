from Prompt import tavily_prompt
from test_neo4j import graph
from datetime import datetime, timedelta

def get_latest_scheduled_date_from_db():
    """Recupera l'ultima data di pubblicazione assoluta presente nel database Neo4j."""
    if not graph:
        return None
    query = """
    MATCH (p:Post)
    WHERE p.date IS NOT NULL
    RETURN p.date AS latest_date
    ORDER BY p.date DESC
    LIMIT 1
    """
    try:
        res = graph.query(query)
        if res and res[0].get("latest_date"):
            return res[0]["latest_date"]
    except Exception as e:
        print(f"Errore query ultima data: {e}")
    return None


def get_next_fixed_publish_date(base_date_str=None):
    """Trova il prossimo giorno utile (0=Lun, 2=Mer, 4=Ven, 6=Dom) partendo da una data base."""
    # 0 = Lunedì, 2 = Mercoledì, 4 = Venerdì, 6 = Domenica
    giorni_pubblicazione = [0, 2, 4, 6]

    if base_date_str:
        base_date = datetime.strptime(base_date_str, "%Y-%m-%d").date()
    else:
        # Fallback alla data odierna solo se il DB è completamente vuoto
        base_date = datetime.now().date()

    next_date = base_date + timedelta(days=1)

    # Scorre i giorni finché non trova un giorno del palinsesto
    while next_date.weekday() not in giorni_pubblicazione:
        next_date += timedelta(days=1)

    return next_date


from langchain_community.tools.tavily_search import TavilySearchResults

# Creiamo l'istanza del tool nativo di LangChain.
# max_results=3 è un buon compromesso per non saturare la context window del LLM
tavily_search_tool = TavilySearchResults(
    max_results=3,
    description=tavily_prompt
)


def save_to_neo4j(title: str, topic: str, claims: list, sources: list, publish_date: str, related_topics: list = None):
    """
    Salva il post, le entità, la data di pubblicazione e genera le relazioni semantiche tra Topic decise dall'AI.
    """
    if not graph:
        print("⚠️ Errore: Database Neo4j non connesso, salto il salvataggio.")
        return

    if related_topics is None:
        related_topics = []

    query = """
    // 1. Crea/Trova il Post e imposta la data
    MERGE (p:Post {title: $title})
    ON CREATE SET p.date = $date
    ON MATCH SET p.date = $date

    // 2. Crea/Trova il Topic principale (normalizzato in minuscolo per sicurezza di indicizzazione)
    MERGE (t:Topic {name: toLower($topic)})
    MERGE (p)-[:COVERS]->(t)

    // 3. Aggiungi le Claims
    WITH p, t
    UNWIND $claims AS claim_text
    MERGE (c:Claim {text: claim_text})
    MERGE (p)-[:EXTRACTS]->(c)
    MERGE (c)-[:RELATED_TO]->(t)

    // 4. Aggiungi le Fonti
    WITH p, t
    UNWIND $sources AS source_name
    MERGE (s:Source {name: source_name})
    MERGE (p)-[:USES]->(s)

    // 5. DINAMICITÀ AI: Generazione relazioni tra topic
    WITH t
    UNWIND $related_topics AS rel_topic_name
    // Cerchiamo nel grafo il topic correlato segnalato dall'LLM
    MATCH (old_t:Topic {name: toLower(rel_topic_name)})
    // Evitiamo che un nodo si colleghi a se stesso
    WHERE old_t <> t
    // Creiamo una relazione bidirezionale o direzionale di correlazione semantica
    MERGE (t)-[:RELATED_TO]->(old_t)
    """

    try:
        graph.query(query, params={
            "title": title,
            "topic": topic,
            "claims": claims,
            "sources": sources,
            "date": publish_date,
            "related_topics": related_topics  # Passato a Cypher
        })
        print("🧠 Knowledge Graph aggiornato con successo con relazioni AI-driven!")
    except Exception as e:
        print(f"❌ Errore durante il salvataggio nel KG: {e}")


def get_covered_context_from_neo4j():
    """
    Estrarre sia i Topic che i singoli Claims dal Knowledge Graph
    usando lo schema esatto definito in save_to_neo4j per evitare warning ed elementi vuoti.
    """
    # Seguiamo esattamente la relazione (c:Claim)-[:RELATED_TO]->(t:Topic)
    # e prendiamo la proprietà c.text definita nel tuo salvataggio.
    query = """
    MATCH (t:Topic)
    OPTIONAL MATCH (c:Claim)-[:RELATED_TO]->(t)
    RETURN t.name AS topic_name, collect(c.text) AS claims
    """

    try:
        records = graph.query(query)

        contesto_kg = []
        for r in records:
            topic = r.get("topic_name")
            # Filtriamo eventuali valori nulli o stringhe vuote
            claims_associati = [claim for claim in r.get("claims", []) if claim]

            info = f"Macro-Topic: '{topic}' (Claims già trattati: {claims_associati})"
            contesto_kg.append(info)

        return contesto_kg if contesto_kg else ["Nessun contenuto precedente nel KG"]
    except Exception as e:
        print(f"⚠️ Errore estrazione dettagliata Cypher: {e}")
        return ["Nessun contenuto precedente nel KG"]


def get_smart_schedule_dates(n_days: int = 0, total_posts: int = 3) -> list[str]:
    """
    Calcola una sequenza di date disponibili garantendo ESCLUSIVAMENTE
    il rispetto del palinsesto fisso (Lun, Mer, Ven, Dom).
    Ignora n_days perché usa le date specifiche.
    """
    from test_neo4j import graph
    from datetime import datetime

    if not graph:
        print("⚠️ Graph non connesso, calcolo date fittizie sul palinsesto.")
        # Fallback basico: partiamo da oggi e cerchiamo il palinsesto
        date_pianificate = []
        base = get_next_fixed_publish_date(datetime.now().strftime("%Y-%m-%d"))
        for _ in range(total_posts):
            date_pianificate.append(base.strftime("%Y-%m-%d"))
            base = get_next_fixed_publish_date(base.strftime("%Y-%m-%d"))
        return date_pianificate

    # 1. Recuperiamo l'ultima data dal DB per avere il punto di partenza
    base_date_str = get_latest_scheduled_date_from_db()

    # 2. Troviamo il primo giorno utile di palinsesto
    giorno_corrente = get_next_fixed_publish_date(base_date_str)

    # Recuperiamo l'occupazione di tutte le date future
    oggi = datetime.now().date()
    query = """
    MATCH (p:Post)
    WHERE p.date >= $oggi
    RETURN p.date AS post_date, count(p) AS count
    """
    try:
        risultati = graph.query(query, params={"oggi": oggi.strftime("%Y-%m-%d")})
        conteggio = {res["post_date"]: res["count"] for res in risultati if res.get("post_date")}
    except Exception as e:
        print(f"Errore query conteggio: {e}")
        conteggio = {}

    date_pianificate = []

    for _ in range(total_posts):
        # Cerca il primo giorno di PALINSESTO con meno di 3 articoli
        while True:
            data_str = giorno_corrente.strftime("%Y-%m-%d")
            occupazione_attuale = conteggio.get(data_str, 0)

            if occupazione_attuale < 3:
                # Trovato! Aggiungiamo alla pianificazione
                date_pianificate.append(data_str)
                # Aggiorniamo virtualmente il dizionario per questo calcolo in blocco
                conteggio[data_str] = occupazione_attuale + 1

                # Il prossimo post dovrà essere schedulato al PROSSIMO giorno di palinsesto
                giorno_corrente = get_next_fixed_publish_date(data_str)
                break
            else:
                # Il giorno è pieno, saltiamo al successivo giorno consentito dal palinsesto
                giorno_corrente = get_next_fixed_publish_date(data_str)

    return date_pianificate


import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



# --- 1. SETUP DEL DATABASE VETTORIALE ---
def setup_vector_database(cartella_documenti="./knowledge_base", db_path="./chroma_db"):
    # Se la cartella non esiste, la crea
    if not os.path.exists(cartella_documenti):
        os.makedirs(cartella_documenti)
        print(f"📁 Cartella '{cartella_documenti}' creata. Inserisci qui i tuoi PDF.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Se il DB esiste già, lo carica senza rifare tutto
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

    # Dividiamo i documenti in frammenti più piccoli (chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documenti)

    # Creiamo e salviamo il vector store locale
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=db_path)
    return vectorstore
