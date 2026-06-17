from datetime import datetime, timedelta

from langchain_core.tools import tool
#from Notion_Stuff import trova_prima_data_disponibile, controlla_disponibilita_data
from test_neo4j import graph
from typing import Optional


@tool
def write_an_article(about: str, author: str, content: str):
    """Write an article about the topic given by user, author is IA"""
    return f"My article about: {about}. \n{content} \n Written by {author}"

@tool
def find_first_available_date_tool(data_partenza: str = None) -> str:
    """
    Trova la prima data disponibile nel Knowledge Graph per pubblicare un articolo.
    Accetta opzionalmente una 'data_partenza' (YYYY-MM-DD). Se non fornita, cerca da oggi.
    Una giornata è piena se ci sono già 3 articoli schedulati.
    """
    if not graph:
        return "Errore: Database Neo4j non connesso."

    if data_partenza == 'null' or not data_partenza:
        oggi = datetime.now().date()
    else:
        oggi = datetime.strptime(data_partenza, "%Y-%m-%d").date()

    # Cerchiamo tutti i Post da 'oggi' in poi e li raggruppiamo per data
    query = """
    MATCH (p:Post)
    WHERE p.date >= $oggi
    RETURN p.date AS post_date, count(p) AS count
    """
    try:
        risultati = graph.query(query, params={"oggi": oggi.strftime("%Y-%m-%d")})
        # Creiamo un dizionario { "YYYY-MM-DD": count }
        conteggio_giornaliero = {res["post_date"]: res["count"] for res in risultati if res.get("post_date")}

        giorno_corrente = oggi
        for _ in range(365):
            data_str = giorno_corrente.strftime("%Y-%m-%d")
            articoli_presenti = conteggio_giornaliero.get(data_str, 0)

            if articoli_presenti < 3:
                return f"La prima data disponibile trovata nel Knowledge Graph è: {data_str}"

            giorno_corrente += timedelta(days=1)

        return "Non ci sono date disponibili nell'arco di un anno."
    except Exception as e:
        return f"Errore durante l'interrogazione del KG: {e}"


@tool
def check_specific_date_tool(data_target: str) -> str:
    """
    Verifica se una data specifica è disponibile nel Knowledge Graph per schedulare un articolo.
    L'argomento 'data_target' deve essere rigorosamente 'YYYY-MM-DD'.
    Una giornata è piena se ci sono già 3 articoli schedulati.
    """
    if not graph:
        return "Errore: Database Neo4j non connesso."

    query = "MATCH (p:Post {date: $date}) RETURN count(p) AS current_count"
    try:
        risultato = graph.query(query, params={"date": data_target})
        current_count = risultato[0]["current_count"] if risultato else 0

        if current_count < 3:
            return (f"La data {data_target} è DISPONIBILE. "
                    f"Attualmente ci sono {current_count} articoli pianificati.")
        else:
            return (f"La data {data_target} è OCCUPATA. "
                    f"Ci sono già {current_count} articoli pianificati (limite massimo).")
    except Exception as e:
        return f"Errore durante l'interrogazione del KG: {e}"

import os
from langchain_community.tools.tavily_search import TavilySearchResults

# Creiamo l'istanza del tool nativo di LangChain.
# max_results=3 è un buon compromesso per non saturare la context window del LLM
tavily_search_tool = TavilySearchResults(
    max_results=3,
    description="""Un motore di ricerca ottimizzato per agenti AI. 
    Usa questo tool per cercare su internet informazioni aggiornate, notizie o per 
    verificare l'accuratezza scientifica e tecnologica di un argomento prima di scriverci un articolo."""
)

from langchain_core.tools import tool


@tool
def get_previous_topics() -> str:
    """
    Usa questo tool per scoprire quali argomenti (Topic) sono già stati trattati nel blog
    così da evitare di proporre ripetizioni.
    """
    if not graph:
        return "Errore: Database Neo4j non connesso."

    # Cypher query per recuperare tutti i nomi dei nodi Topic
    query = "MATCH (t:Topic) RETURN t.name AS topic_name"

    try:
        risultati = graph.query(query)
        if not risultati:
            return "Nessun topic trattato finora. Il blog è vuoto."

        lista_topic = [res["topic_name"] for res in risultati]
        return f"I topic già trattati nel blog sono: {', '.join(lista_topic)}."
    except Exception as e:
        return f"Errore durante l'interrogazione del KG: {e}"


@tool
def get_topic_claims(topic_name: str) -> str:
    """
    Usa questo tool durante la stesura dell'articolo per recuperare le affermazioni chiave (Claims)
    fatte in passato su un determinato topic (topic_name), così da mantenere coerenza.
    """
    if not graph:
        return "Errore: Database Neo4j non connesso."

    # Cypher query per recuperare le Claim collegate a un Topic specifico (ignorando maiuscole/minuscole)
    query = """
    MATCH (c:Claim)-[:RELATED_TO]->(t:Topic)
    WHERE toLower(t.name) = toLower($topic)
    RETURN c.text AS claim_text
    """

    try:
        risultati = graph.query(query, params={"topic": topic_name})
        if not risultati:
            return f"Non ho trovato affermazioni precedenti sul topic '{topic_name}'."

        lista_claims = [res["claim_text"] for res in risultati]
        return f"Affermazioni passate su {topic_name}:\n- " + "\n- ".join(lista_claims)
    except Exception as e:
        return f"Errore durante l'interrogazione del KG: {e}"


# Aggiungi in Tools.py (assicurati che non abbia @tool sopra)

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


def get_smart_schedule_dates(n_days: int, total_posts: int = 3) -> list[str]:
    """
    Calcola una sequenza di date disponibili, garantendo che ci siano
    almeno 'n_days' tra una pubblicazione e l'altra, saltando i giorni pieni.
    """
    from test_neo4j import graph  # Assicurati di importare l'istanza del tuo graph

    if not graph:
        print("⚠️ Graph non connesso, fallback date sequenziali.")
        # Fallback basico se Neo4j è spento
        base = datetime.now().date()
        return [(base + timedelta(days=i * n_days)).strftime("%Y-%m-%d") for i in range(total_posts)]

    oggi = datetime.now().date()

    # Recuperiamo l'occupazione di tutte le date future
    query = """
    MATCH (p:Post)
    WHERE p.date >= $oggi
    RETURN p.date AS post_date, count(p) AS count
    """
    try:
        risultati = graph.query(query, params={"oggi": oggi.strftime("%Y-%m-%d")})
        # Mappa { "2026-06-18": 3, "2026-06-20": 1, ... }
        conteggio = {res["post_date"]: res["count"] for res in risultati if res.get("post_date")}
    except Exception as e:
        print(f"Errore query conteggio: {e}")
        conteggio = {}

    date_pianificate = []
    giorno_corrente = oggi

    for _ in range(total_posts):
        # Cerca il primo giorno con meno di 3 articoli a partire da 'giorno_corrente'
        while True:
            data_str = giorno_corrente.strftime("%Y-%m-%d")
            occupazione_attuale = conteggio.get(data_str, 0)

            if occupazione_attuale < 3:
                # Trovato! Aggiungiamo alla pianificazione
                date_pianificate.append(data_str)
                # Aggiorniamo virtualmente il dizionario per non sovraccaricare il giorno
                # se l'LLM dovesse pianificare più post nello stesso giorno (se n_days=0)
                conteggio[data_str] = occupazione_attuale + 1

                # Il prossimo post dovrà essere schedulato almeno 'n_days' dopo questo
                giorno_corrente += timedelta(days=n_days)
                break
            else:
                # Giorno pieno, proviamo il giorno successivo
                giorno_corrente += timedelta(days=1)

    return date_pianificate


import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Usa OpenAIEmbeddings se preferisci
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool


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


from langchain_core.tools import tool

# --- 2. CREAZIONE DEL TOOL PER L'AGENTE (Versione con DEBUG) ---
vectorstore_db = setup_vector_database()

if vectorstore_db:
    # Impostiamo il retriever per restituire i 3 chunk più rilevanti
    retriever = vectorstore_db.as_retriever(search_kwargs={"k": 3})


    @tool
    def rag_document_retriever(query: str) -> str:
        """Usa questo tool per cercare informazioni, definizioni e concetti approfonditi all'interno dei documenti e manuali locali del blog. Restituisce frammenti di testo da usare come citazioni per supportare l'articolo."""

        # 🟢 ECCO LA SPIA! Questa riga stamperà in rosso/visibile sul terminale l'uso del RAG
        print(f"\n📚 [DEBUG RAG] L'agente ha attivato il tool sui PDF locali con la query: '{query}'\n")

        # Facciamo la vera ricerca vettoriale
        documenti_trovati = retriever.invoke(query)

        if not documenti_trovati:
            return "Nessuna informazione rilevante trovata nei PDF."

        # Uniamo i frammenti trovati in un unico testo da passare all'LLM
        testo_risultati = "\n\n--- FRAMMENTO --- \n".join([doc.page_content for doc in documenti_trovati])
        return testo_risultati

else:
    # Fallback se non ci sono PDF
    @tool
    def rag_document_retriever(query: str) -> str:
        """Tool fittizio per quando non ci sono documenti"""
        return "Nessun documento locale disponibile nel database vettoriale."