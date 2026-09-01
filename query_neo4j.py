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


def get_save_post_to_neo4j_query_old() -> str:
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

    // 5. DINAMICITÀ AI: Generazione relazioni arricchite tra topic
    WITH t
    UNWIND $related_topics AS rel
    // rel ora è un dizionario con target_topic, relationship_type, reason
    MATCH (old_t:Topic {name: toLower(rel.target_topic)})
    WHERE old_t <> t
    // Creiamo la relazione e iniettiamo le proprietà
    MERGE (t)-[r:RELATED_TO]->(old_t)
    ON CREATE SET r.type = rel.relationship_type, r.reason = rel.reason
    ON MATCH SET r.type = rel.relationship_type, r.reason = rel.reason
    """

def get_save_post_to_neo4j_query() -> str:
    """Restituisce la query per salvare un nuovo post, topic, claims, fonti, documentazione e relazioni nel Knowledge Graph."""
    return """
    // 1. Crea/Trova il Post e imposta la data
    MERGE (p:Post {title: $title})
    ON CREATE SET p.date = $date
    ON MATCH SET p.date = $date

    // 2. Crea/Aggiorna il nodo Documentation (l'articolo completo)
    // Facciamo il MERGE direttamente dal Post per evitare costosi calcoli di matching su testi molto lunghi
    MERGE (p)-[:USES]->(d:Documentation)
    ON CREATE SET d.text = $documentation_text
    ON MATCH SET d.text = $documentation_text

    // 3. Crea/Trova il Topic principale (normalizzato in minuscolo per sicurezza di indicizzazione)
    MERGE (t:Topic {name: toLower($topic)})
    MERGE (p)-[:COVERS]->(t)

    // 4. Aggiungi le Claims
    WITH p, t
    UNWIND $claims AS claim_text
    MERGE (c:Claim {text: claim_text})
    MERGE (p)-[:EXTRACTS]->(c)
    MERGE (c)-[:RELATED_TO]->(t)

    // 5. Aggiungi le Fonti
    WITH p, t
    UNWIND $sources AS source_name
    MERGE (s:Source {name: source_name})
    MERGE (p)-[:USES]->(s)

    // 6. DINAMICITÀ AI: Generazione relazioni arricchite tra topic
    WITH t
    UNWIND $related_topics AS rel
    // rel ora è un dizionario con target_topic, relationship_type, reason
    MATCH (old_t:Topic {name: toLower(rel.target_topic)})
    WHERE old_t <> t
    // Creiamo la relazione e iniettiamo le proprietà
    MERGE (t)-[r:RELATED_TO]->(old_t)
    ON CREATE SET r.type = rel.relationship_type, r.reason = rel.reason
    ON MATCH SET r.type = rel.relationship_type, r.reason = rel.reason
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


def get_post_for_evaluation_query() -> str:
    """
        Estrae i post recenti dal Knowledge Graph completi di topic, claims e fonti
        per passarli al nodo o allo script di valutazione.
    """

    query = """
        MATCH (p:Post)-[:COVERS_TOPIC]->(t:Topic)
        OPTIONAL MATCH (p)-[:HAS_CLAIM]->(c:Claim)
        OPTIONAL MATCH (p)-[:USES_SOURCE]->(s:Source)
        RETURN p.title AS title, 
               p.date AS publish_date, 
               t.name AS topic, 
               collect(DISTINCT c.content) AS claims, 
               collect(DISTINCT s.url) AS sources
        ORDER BY p.date DESC
        LIMIT $limit
        """
    return query


def get_enhanced_topic_context_query() -> str:
    """
    Recupera i claim diretti di un topic e tutte le sue relazioni arricchite
    con gli altri topic (tipologia di arco e motivazione semantica).
    """
    return """
    MATCH (t:Topic)
    WHERE toLower(t.name) = toLower($topic)

    // 1. Recupera i claim storici associati a questo topic
    OPTIONAL MATCH (c:Claim)-[:RELATED_TO]->(t)

    // 2. Recupera i topic correlati e i metadati dell'arco indotti dall'AI
    OPTIONAL MATCH (t)-[r:RELATED_TO]->(other:Topic)

    RETURN 
        t.name AS topic_name,
        collect(distinct c.text) AS direct_claims,
        collect(distinct {target: other.name, type: r.type, reason: r.reason}) AS relations
    """