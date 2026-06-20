from test_neo4j import graph

def get_posts_for_evaluation(limit: int = 5):
    """
    Estrae i post recenti dal Knowledge Graph completi di topic, claims e fonti
    per passarli al nodo o allo script di valutazione.
    """
    if not graph:
        print("⚠️ Errore: Database Neo4j non connesso.")
        return []

    # Sostituisci i nomi delle label e delle relazioni (Post, Topic, Claim, Source, HAS_CLAIM, ecc.)
    # con quelli esatti che hai usato nella tua get_save_post_to_neo4j_query()
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

    try:
        results = graph.query(query, params={"limit": limit})
        print(f"✅ Recuperati {len(results)} post per la valutazione.")
        return results
    except Exception as e:
        print(f"❌ Errore durante l'estrazione dal KG: {e}")
        return []