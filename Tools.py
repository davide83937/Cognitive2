from datetime import datetime, timedelta
from langchain_core.tools import tool


from function_tool import setup_vector_database, get_latest_scheduled_date_from_db, get_next_fixed_publish_date, \
    getRetriever, tavily_search_tool
from query_neo4j import get_post_count_by_date_query, get_all_topics_query, get_claims_by_topic_query
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

    query = get_post_count_by_date_query()

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
    query = get_all_topics_query()

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
    from test_neo4j import graph  # Assicurati di avere l'import
    if not graph:
        return "Errore: Database Neo4j non connesso."

    from query_neo4j import get_claims_by_topic_query
    query = get_claims_by_topic_query()

    print(f"\n🧠 [DEBUG KG] L'agente sta interrogando la memoria per il topic: '{topic_name}'")

    try:
        risultati = graph.query(query, params={"topic": topic_name})
        if not risultati:
            print(f"❌ [DEBUG KG] Nessuna informazione precedente trovata. Argomento nuovo.")
            return f"Non ho trovato affermazioni precedenti sul topic '{topic_name}'."

        lista_claims = [res["claim_text"] for res in risultati]
        risultato_testuale = f"Affermazioni passate su {topic_name}:\n- " + "\n- ".join(lista_claims)

        print(f"✅ [DEBUG KG] Trovate {len(lista_claims)} informazioni storiche passate al RAG/LLM.")
        return risultato_testuale
    except Exception as e:
        return f"Errore durante l'interrogazione del KG: {e}"

@tool
def rag_document_retriever(query: str) -> str:
    """Usa questo tool per cercare informazioni, definizioni e concetti approfonditi all'interno dei documenti e manuali locali del blog.
    Restituisce frammenti di testo da usare come citazioni per supportare l'articolo."""

    retriever = getRetriever()
    # 4. Gestisci il fallback internamente se il DB non è stato caricato
    if not retriever:
        return "Nessun documento locale disponibile nel database vettoriale. Prosegui senza citazioni dai PDF."

    # 🟢 DEBUG VISIVO
    print(f"\n📚 [DEBUG RAG] L'agente ha attivato il tool sui PDF locali con la query: '{query}'\n")

    # 5. Ricerca vettoriale
    documenti_trovati = retriever.invoke(query)

    if not documenti_trovati:
        return "Nessuna informazione rilevante trovata nei PDF per questa query."

    # 6. Unione dei risultati
    testo_risultati = "\n\n--- FRAMMENTO --- \n".join([doc.page_content for doc in documenti_trovati])
    return testo_risultati


@tool
def verified_internet_search(query: str) -> str:
    """
    Usa questo tool per cercare su internet. Cerca le informazioni e valuta
    dinamicamente l'affidabilità di ogni fonte trovata.
    """
    print(f"\n🌐 [DEBUG TAVILY] L'agente ha cercato sul web: '{query}'")
    risultati_grezzi = tavily_search_tool.invoke({"query": query})

    from Models import get_llm
    llm_giudice = get_llm()

    prompt_giudice = f"""
    Sei un revisore di fonti esperto. Ecco i risultati di una ricerca web:
    {risultati_grezzi}

    Il tuo compito è analizzare l'URL e il contenuto di ciascun risultato. 
    Seleziona e restituisci SOLO i risultati che provengono da fonti attendibili 
    (testate giornalistiche, siti istituzionali, portali scientifici/accademici o blog tecnici riconosciuti).
    Scarta tutto ciò che sembra un forum, un social network o un sito promozionale di bassa qualità.

    Rispondi fornendo un breve riepilogo delle fonti salvate e il loro contenuto utile. E spiega brevemente quali hai scartato e perché.
    Se nessuna fonte è buona, scrivi "Nessuna fonte affidabile trovata".
    """

    risposta_filtrata = llm_giudice.invoke([{"role": "user", "content": prompt_giudice}])

    # AGGIUNGIAMO QUESTO PRINT PER VEDERE IL RAGIONAMENTO DEL GIUDICE
    print(f"\n⚖️ [DEBUG GIUDICE FONTI] Verdetto sulle fonti trovate:\n{risposta_filtrata.content}\n")

    return risposta_filtrata.content


from langchain_core.tools import tool

from test_neo4j import graph


@tool
def intelligent_topic_matcher(new_topic: str) -> str:
    """
    Usa questo tool per scoprire SE e COME un nuovo argomento è già stato trattato nel Knowledge Graph.
    Fornisce la corrispondenza semantica esatta del nome del Topic nel database.
    """
    from Models import get_llm
    if not graph:
        return "Errore: Database Neo4j non connesso."

    # 1. Estraiamo TUTTI i Topic esistenti nel database (e magari qualche claim chiave)
    # Se il DB è piccolo/medio, possiamo portarli tutti in memoria.
    query = """
    MATCH (t:Topic)
    OPTIONAL MATCH (c:Claim)-[:RELATED_TO]->(t)
    RETURN t.name AS topic_name, collect(c.text)[0..2] AS sample_claims
    """
    try:
        records = graph.query(query)
        if not records:
            return f"Il database è vuoto. '{new_topic}' è un argomento 100% nuovo."

        lista_esistenti = []
        for r in records:
            nome = r['topic_name']
            claims = r['sample_claims']
            lista_esistenti.append(f"- Topic: '{nome}' (Esempi trattati: {claims})")

        contesto_db = "\n".join(lista_esistenti)

    except Exception as e:
        return f"Errore durante l'interrogazione di Neo4j: {e}"

    # 2. Usiamo un LLM "Giudice" per fare il matching semantico
    llm_giudice = get_llm()
    prompt_giudice = f"""
    Sei un analista semantico. Il tuo compito è confrontare un NUOVO argomento con una lista di argomenti GIA' ESISTENTI nel database.

    NUOVO ARGOMENTO PROPOSTO: "{new_topic}"

    ARGOMENTI ESISTENTI NEL DATABASE:
    {contesto_db}

    DOMANDA: Il nuovo argomento proposto è concettualmente uguale, o una sotto-categoria molto stretta, di uno degli argomenti esistenti?
    Considera sinonimi, acronimi (es. 5G e Rete 5G) e concetti correlati.

    REGOLE DI RISPOSTA (IMPORTANTISSIMO):
    - Se C'E' una corrispondenza semantica, rispondi SOLO con il NOME ESATTO del Topic esistente (copialo identico a come è scritto nel database). Non aggiungere altre parole.
    - Se NON C'E' nessuna corrispondenza (è un argomento totalmente nuovo), rispondi ESATTAMENTE con la parola: "NESSUNO".
    """

    risposta_llm = llm_giudice.invoke([{"role": "user", "content": prompt_giudice}]).content.strip()

    # 3. Formattiamo la risposta per l'agente principale
    if risposta_llm == "NESSUNO":
        return f"Nessuna corrispondenza semantica trovata. L'argomento '{new_topic}' è nuovo."
    else:
        return f"Trovata corrispondenza semantica! Nel database l'argomento è salvato ESATTAMENTE con il nome: '{risposta_llm}'. Usa questo nome per interrogare i claims."