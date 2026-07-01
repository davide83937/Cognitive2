import requests
import os

# Sostituisci con l'IP e la porta esatti esposti dal tuo server Flask
# Dai log del tuo server vedo che è in ascolto su: http://192.168.1.2:5000
API_URL = os.environ.get("CYPHER_API_URL", "http://192.168.1.2:5000/generate_cypher")


def generate_cypher(query_testuale: str) -> str:
    """
    Genera una query Cypher deterministica delegando il calcolo al server API (Flask)
    che ospita il modello fine-tuned Qwen-7B sulla RTX 3070.
    """
    # 1. Manteniamo il prompt di sistema rigido per istruire correttamente il modello
    prompt_text = """Genera SOLO la query Neo4j Cypher. Non aggiungere spiegazioni o formattazione markdown.
    REGOLE RIGIDE:
    1. Restituisci in RETURN SOLO le variabili esplicitamente richieste.
    2. Le date ('date') appartengono ESCLUSIVAMENTE al nodo Post (p.date). I Claim non hanno date.
    3. Se viene richiesto di estrarre i testi dei claim o il contesto trattato filtrando per data, devi unire esplicitamente tutti e tre i nodi nel MATCH principale: (c:Claim)-[:RELATED_TO]->(t:Topic)<-[:COVERS]-(p:Post). Non inventare sotto-query o comprehension nel RETURN.
    4. Mantieni i percorsi ottimali e diretti.

   SCHEMA CONSENTITO:
- Nodi: Post {title, date}, Topic {name}, Claim {text}, Source {name}
- Relazioni Post: (Post)-[:COVERS]->(Topic), (Post)-[:EXTRACTS]->(Claim), (Post)-[:USES]->(Source)
- Relazioni Topic: (Topic)-[r:RELATED_TO]->(Topic) dove r.type DEVE ESSERE 'PREREQUISITO', 'CONTRASTO', 'ESTENSIONE', 'SIMILARE', 'SOTTO_CATEGORIA' o 'APPLICAZIONE'.

    ESEMPI DI RIFERIMENTO:
    Utente: Seleziona i post di aprile 2026 che coprono il topic 'Docker'.
    Cypher: MATCH (p:Post)-[:COVERS]->(t:Topic) WHERE p.date STARTS WITH '2026-04' AND t.name =~ '(?i).*docker.*' RETURN p.title

    Utente: Restituisci il nome di tutti i topic presenti e i testi dei claim estratti negli ultimi 30 giorni.
    Cypher: MATCH (c:Claim)-[:RELATED_TO]->(t:Topic)<-[:COVERS]-(p:Post) WHERE p.date >= '2026-06-01' RETURN DISTINCT t.name, c.text ORDER BY t.name ASC

    Utente: Topic simili a 'RAG' senza contrasti.
    Cypher: MATCH (t1:Topic)-[:RELATED_TO {type: 'SIMILARE'}]->(t2:Topic) WHERE t2.name =~ '(?i).*rag.*' AND NOT (t1)-[:RELATED_TO {type: 'CONTRASTO'}]-(:Topic) RETURN t1.name

    Richiesta: """

    # 2. Assembliamo il prompt completo (Istruzioni + Richiesta NL)
    full_prompt = prompt_text + query_testuale + "\nCypher: "

    # 3. Invio della richiesta HTTP POST al server Flask
    try:
        response = requests.post(
            API_URL,
            json={"prompt": full_prompt},
            timeout=45  # Timeout generoso nel caso la GPU impieghi qualche secondo
        )
        response.raise_for_status()  # Verifica che non ci siano errori HTTP (es. 500 o 404)

        # 4. Estrazione della query ripulita restituita dal server
        data = response.json()
        risultato_pulito = data.get("cypher_query", "")

        return risultato_pulito

    except requests.exceptions.RequestException as e:
        print(f"❌ [API ERROR] Errore di comunicazione con il server Text-to-Cypher: {e}")
        # Restituiamo una stringa d'errore che l'agente può intercettare o loggare
        return f"Errore Cypher (API inattiva): {e}"