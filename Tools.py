from langchain_core.tools import tool
from Notion_Stuff import trova_prima_data_disponibile, controlla_disponibilita_data
from test_neo4j import graph
from typing import Optional


@tool
def write_an_article(about: str, author: str, content: str):
    """Write an article about the topic given by user, author is IA"""
    return f"My article about: {about}. \n{content} \n Written by {author}"


@tool
def find_first_available_date_tool(data_partenza: Optional[str] = None) -> str:
    """
    Trova la prima data disponibile su Notion per pubblicare un articolo.
    Accetta opzionalmente una 'data_partenza' (YYYY-MM-DD). Se non fornita, cerca da oggi.
    Restituisce la data trovata o un messaggio se il calendario è pieno.
    Una giornata è piena se ci sono già 3 articoli schedulati
    """
    # Se il modello passa la stringa 'null' o un valore nullo, forziamolo a None
    if data_partenza == 'null':
        data_partenza = None

    data_trovata = trova_prima_data_disponibile(data_partenza)
    if data_trovata:
        return f"La prima data disponibile trovata su Notion è: {data_trovata}"
    else:
        return "Non ci sono date disponibili su Notion per l'arco di tempo richiesto."


@tool
def check_specific_date_tool(data_target: str) -> str:
    """
    Verifica se una data specifica è disponibile su Notion per schedulare un articolo.
    L'argomento 'data_target' deve essere rigorosamente nel formato stringa 'YYYY-MM-DD' (es. '2026-06-01').
    Da usare quando l'utente propone una data precisa o chiede se un determinato giorno è libero.
    Una giornata è piena se ci sono già 3 articoli schedulati.
    """
    risultato = controlla_disponibilita_data(data_target)

    if risultato is None:
        return f"Impossibile verificare la disponibilità per la data {data_target} a causa di un errore di comunicazione con Notion."

    if risultato["is_available"]:
        return (f"La data {data_target} è DISPONIBILE su Notion. "
                f"Attualmente ci sono {risultato['current_count']} articoli pianificati per questo giorno.")
    else:
        return (f"La data {data_target} è OCCUPATA/PIENA. "
                f"Ci sono già {risultato['current_count']} articoli pianificati, raggiungendo il limite massimo.")



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

def save_to_neo4j(title: str, topic: str, claims: list, sources: list):
    """
    Salva il post approvato e le sue entità nel Knowledge Graph Neo4j,
    creando tutte le relazioni necessarie.
    """
    if not graph:
        print("⚠️ Errore: Database Neo4j non connesso, salto il salvataggio nel KG.")
        return

    # Usiamo MERGE in Cypher, che crea il nodo/relazione solo se non esiste già
    query = """
    // 1. Crea/Trova il Post e il Topic
    MERGE (p:Post {title: $title})
    MERGE (t:Topic {name: toLower($topic)})
    MERGE (p)-[:COVERS]->(t)

    // 2. Aggiungi le Claims (usiamo UNWIND per iterare sulla lista)
    WITH p, t
    UNWIND $claims AS claim_text
    MERGE (c:Claim {text: claim_text})
    MERGE (p)-[:EXTRACTS]->(c)
    MERGE (c)-[:RELATED_TO]->(t)

    // 3. Aggiungi le Fonti
    WITH p
    UNWIND $sources AS source_name
    MERGE (s:Source {name: source_name})
    MERGE (p)-[:USES]->(s)
    """

    try:
        graph.query(query, params={
            "title": title,
            "topic": topic,
            "claims": claims,
            "sources": sources
        })
        print("🧠 Knowledge Graph aggiornato con successo!")
    except Exception as e:
        print(f"❌ Errore durante il salvataggio nel KG: {e}")