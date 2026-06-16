import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

# Carica le variabili dal file .env
load_dotenv()

print("Tentativo di connessione a Neo4j Aura in corso...")

try:
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        refresh_schema=False
    )

    # Eseguiamo una query di test semplicissima
    risultato = graph.query("RETURN 'Connessione Riuscita!' AS messaggio")
    print("✅", risultato[0]['messaggio'])

except Exception as e:
    print(f"❌ Errore di connessione: {e}")