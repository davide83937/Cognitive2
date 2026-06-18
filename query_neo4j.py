def get_all_topics_query() -> str:
    """Restituisce la query per ottenere tutti i nomi dei Topic."""
    return "MATCH (t:Topic) RETURN t.name AS topic_name"


def get_post_count_by_date_query() -> str:
    """Restituisce la query per contare quanti post ci sono in una data specifica."""
    return "MATCH (p:Post {date: $date}) RETURN count(p) AS current_count"


def get_claims_by_topic_query() -> str:
    """Restituisce la query per ottenere le affermazioni (Claims) collegate a un Topic."""
    return """
    MATCH (c:Claim)-[:RELATED_TO]->(t:Topic)
    WHERE toLower(t.name) = toLower($topic)
    RETURN c.text AS claim_text
    """

def get_latest_scheduled_date_query() -> str:
    """Restituisce la query per ottenere l'ultima data di pubblicazione assoluta presente nel database."""
    return """
    MATCH (p:Post)
    WHERE p.date IS NOT NULL
    RETURN p.date AS latest_date
    ORDER BY p.date DESC
    LIMIT 1
    """


def get_save_post_to_neo4j_query() -> str:
    """Restituisce la query per salvare un nuovo post, topic, claims, fonti e relazioni nel Knowledge Graph."""
    return """
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

def get_covered_context_query() -> str:
    """Restituisce la query per estrarre tutti i Topic e i singoli Claims già trattati."""
    return """
    MATCH (t:Topic)
    OPTIONAL MATCH (c:Claim)-[:RELATED_TO]->(t)
    RETURN t.name AS topic_name, collect(c.text) AS claims
    """


def get_future_post_counts_query() -> str:
    """Restituisce la query per ottenere l'occupazione di tutte le date future a partire da un giorno specifico."""
    return """
    MATCH (p:Post)
    WHERE p.date >= $oggi
    RETURN p.date AS post_date, count(p) AS count
    """