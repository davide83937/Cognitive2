from datetime import datetime, timedelta
from langchain_core.tools import tool
from function_tool import setup_vector_database, get_latest_scheduled_date_from_db, get_next_fixed_publish_date
from test_neo4j import graph


@tool
def write_an_article(about: str, author: str, content: str):
    """Write an article about the topic given by user, author is IA"""
    return f"My article about: {about}. \n{content} \n Written by {author}"




@tool
def find_first_available_date_tool(data_partenza: str = None) -> str:
    """
    Trova la prima data disponibile nel Knowledge Graph per pubblicare un articolo.
    La programmazione avviene rigorosamente solo di Lunedì, Mercoledì, Venerdì e Domenica.
    Se 'data_partenza' non è fornita, riparte dall'ultimo articolo pubblicato nel database.
    Una giornata è piena se ci sono già 3 articoli schedulati.
    """
    if not graph:
        return "Errore: Database Neo4j non connesso."

    # Se l'LLM non ha una data specifica di partenza, peschiamo l'ultima assoluta dal DB
    if not data_partenza or data_partenza == 'null':
        base_date_str = get_latest_scheduled_date_from_db()
    else:
        base_date_str = data_partenza

    # Troviamo il prossimo giorno consentito dal palinsesto
    giorno_corrente = get_next_fixed_publish_date(base_date_str)

    query = "MATCH (p:Post {date: $date}) RETURN count(p) AS current_count"

    # Cerchiamo in avanti saltando ai soli giorni consentiti
    for _ in range(50):  # Limite di tentativi per non creare loop infiniti
        data_str = giorno_corrente.strftime("%Y-%m-%d")

        try:
            risultato = graph.query(query, params={"date": data_str})
            articoli_presenti = risultato[0]["current_count"] if risultato else 0

            if articoli_presenti < 3:
                return f"La prima data disponibile in palinsesto (Lun, Mer, Ven, Dom) nel Knowledge Graph è: {data_str}"

            # Se il giorno è pieno, saltiamo al PROSSIMO giorno di palinsesto
            giorno_corrente = get_next_fixed_publish_date(data_str)

        except Exception as e:
            return f"Errore durante l'interrogazione del KG: {e}"

    return "Non ci sono date disponibili nel palinsesto."


@tool
def check_specific_date_tool(data_target: str) -> str:
    """
    Verifica se una data specifica è disponibile nel Knowledge Graph per schedulare un articolo.
    L'argomento 'data_target' deve essere rigorosamente 'YYYY-MM-DD'.
    Verifica anche che la data rispetti il palinsesto: Lunedì, Mercoledì, Venerdì, Domenica.
    Una giornata è piena se ci sono già 3 articoli schedulati.
    """
    if not graph:
        return "Errore: Database Neo4j non connesso."

    # 1. Verifica che il giorno della settimana sia consentito
    giorni_pubblicazione = [0, 2, 4, 6]
    giorni_nomi = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]

    try:
        target_date_obj = datetime.strptime(data_target, "%Y-%m-%d").date()
    except ValueError:
        return "Errore: Formato data non valido, usa YYYY-MM-DD."

    giorno_settimana = target_date_obj.weekday()

    if giorno_settimana not in giorni_pubblicazione:
        return (f"Rifiutato: La data {data_target} è un {giorni_nomi[giorno_settimana]}. "
                f"Il blog pubblica solo di Lunedì, Mercoledì, Venerdì e Domenica. "
                f"Prova a cercare la prima data disponibile.")

    # 2. Se è un giorno consentito, verifica l'occupazione in Neo4j
    query = "MATCH (p:Post {date: $date}) RETURN count(p) AS current_count"
    try:
        risultato = graph.query(query, params={"date": data_target})
        current_count = risultato[0]["current_count"] if risultato else 0

        if current_count < 3:
            return (f"La data {data_target} ({giorni_nomi[giorno_settimana]}) è DISPONIBILE in palinsesto. "
                    f"Attualmente ci sono {current_count} articoli pianificati.")
        else:
            return (f"La data {data_target} è OCCUPATA. "
                    f"Ci sono già {current_count} articoli pianificati (limite massimo).")
    except Exception as e:
        return f"Errore durante l'interrogazione del KG: {e}"





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


