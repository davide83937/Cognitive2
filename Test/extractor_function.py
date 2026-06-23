from test_neo4j import graph
from query_neo4j import get_post_for_evaluation_query

def get_posts_for_evaluation(limit: int = 5):
    """
    Estrae i post recenti dal Knowledge Graph completi di topic, claims e fonti
    per passarli al nodo o allo script di valutazione.
    """
    if not graph:
        print("⚠️ Errore: Database Neo4j non connesso.")
        return []

    query = get_post_for_evaluation_query()

    try:
        results = graph.query(query, params={"limit": limit})
        print(f"✅ Recuperati {len(results)} post per la valutazione.")
        return results
    except Exception as e:
        print(f"❌ Errore durante l'estrazione dal KG: {e}")
        return []