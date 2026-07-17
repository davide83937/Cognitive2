import os
import json
import time
import requests
from neo4j import GraphDatabase, RoutingControl
from neo4j.exceptions import Neo4jError

# ==========================================
# CONFIGURAZIONE PARAMETRI
# ==========================================
DATASET_PATH = "test_set_esterno.json"  # Nuovo file del test set

# L'API Flask in locale funziona (Lascia invariato)
API_URL = os.environ.get("CYPHER_API_URL", "http://192.168.1.2:5000/generate_cypher")

# --- MODIFICA QUESTA PARTE PER IL CLOUD ---
# Se usi Neo4j Aura, il prefisso è solitamente "neo4j+s://"
NEO4J_URI = "neo4j+s://13ee89a5.databases.neo4j.io"
NEO4J_USER = "13ee89a5" # o il tuo username cloud
NEO4J_PASSWORD = "Q_hewmVIUQ_Vm1vxnS-BxiOSE6Gvn30cddRuqFSnS7Y"
# ------------------------------------------

# ==========================================
# FUNZIONI DI GENERAZIONE E VALIDAZIONE
# ==========================================
def load_test_dataset(filepath):
    """Carica tutti gli elementi dal dataset JSON (supporta sia Array che JSONL)."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        try:
            # 1. Prova a leggerlo come un JSON standard (es. un array [{}, {}])
            data = json.loads(content)
        except json.JSONDecodeError:
            # 2. Se fallisce, assume che sia un file JSON Lines (un oggetto per riga)
            for riga in content.split('\n'):
                if riga.strip():  # Ignora eventuali righe vuote
                    data.append(json.loads(riga))

    print(f"📦 Dataset caricato con successo. Elementi totali: {len(data)}")
    return data


def generate_cypher(query_testuale: str) -> str:
    """
    Genera una query Cypher deterministica delegando il calcolo al server API (Flask)
    che ospita il modello fine-tuned Qwen-7B sulla RTX 3070.
    """
    prompt_text_old = """Genera SOLO la query Neo4j Cypher. Non aggiungere spiegazioni.
        REGOLE RIGIDE:
        1. Restituisci in RETURN SOLO le variabili esplicitamente richieste.
        2. Le date ('date') appartengono ESCLUSIVAMENTE al nodo Post (p.date).
        3. Usa SEMPRE le regex case-insensitive per cercare i nomi (es. `=~ '(?i).*nome.*'`).
        4. I Tipi di relazione devono essere TUTTI IN MAIUSCOLO (es. 'APPLICAZIONE', 'CONTRASTO').
        5. Per accedere alle proprietà delle relazioni, assegna una variabile nel MATCH (es. `-[r:RELATED_TO]->`) e usala nel RETURN (es. `r.reason`).
        6. Non usare MAI due clausole WHERE nello stesso MATCH. Usa AND.
        7. Usa ESCLUSIVAMENTE i nodi e le relazioni elencate nello SCHEMA CONSENTITO. Non inventare relazioni come [:CONTRO].
        8. Se la richiesta menziona nodi multipli (es. claim e documenti), DEVI collegarli TUTTI esplicitamente usando clausole MATCH (es. MATCH (p)-[:EXTRACTS]->(c:Claim) MATCH (p)-[:USES]->(d:Documentation)) prima di usarli nel RETURN.
        9. Usa ESCLUSIVAMENTE le proprietà dichiarate nello SCHEMA CONSENTITO. Non inventare proprietà come p.content o p.url.


        SCHEMA CONSENTITO:
        - Nodi: Post {title, date}, Topic {name}, Claim {text}, Source {name}, Documentation {text}
        - Relazioni Post: (Post)-[:COVERS]->(Topic), (Post)-[:USES]->(Documentation), (Post)-[:EXTRACTS]->(Claim)
        - Relazioni Topic: (Topic)-[r:RELATED_TO]->(Topic) dove r.type DEVE ESSERE 'PREREQUISITO', 'CONTRASTO', 'ESTENSIONE', 'SIMILARE', 'SOTTO_CATEGORIA' o 'APPLICAZIONE'. La relazione 'r' possiede la proprietà 'reason'.

        ESEMPI DI RIFERIMENTO:
        Utente: Seleziona i post di aprile 2026 che coprono il topic 'Docker'.
        Cypher: MATCH (p:Post)-[:COVERS]->(t:Topic) WHERE p.date STARTS WITH '2026-04' AND t.name =~ '(?i).*docker.*' RETURN p.title
        Utente: Topic simili a 'RAG' senza contrasti.
        Cypher: MATCH (t1:Topic)-[:RELATED_TO {type: 'SIMILARE'}]->(t2:Topic) WHERE t2.name =~ '(?i).*rag.*' AND NOT (t1)-[:RELATED_TO {type: 'CONTRASTO'}]-(:Topic) RETURN t1.name
        Utente: Trova i topic che sono un'estensione di 'Python' e restituisci anche il motivo.
        Cypher: MATCH (t1:Topic)-[r:RELATED_TO {type: 'ESTENSIONE'}]->(t2:Topic) WHERE t2.name =~ '(?i).*python.*' RETURN t1.name, r.reason

        Richiesta: """


    prompt_text1 = """
    Genera SOLO la query Neo4j Cypher. Non aggiungere spiegazioni o formattazione markdown.

REGOLE RIGIDE:
1. Restituisci in RETURN SOLO le variabili esplicitamente richieste.
2. Le date ('date') appartengono ESCLUSIVAMENTE al nodo Post (p.date).
3. Usa SEMPRE le regex case-insensitive per cercare i nomi o i testi (es. `=~ '(?i).*nome.*'`).
4. I Tipi di relazione devono essere TUTTI IN MAIUSCOLO: COVERS, EXTRACTS, RELATED_TO, USES.
5. Se la richiesta menziona nodi multipli, DEVI collegarli esplicitamente tramite i path corretti.
6. Non usare mai due clausole WHERE separate in un MATCH; usa l'operatore AND.

SCHEMA CONSENTITO:
- Nodi: Post {title, date}, Topic {name}, Claim {text}, Source {name}
- Relazioni:
  (Post)-[:COVERS]->(Topic)
  (Post)-[:EXTRACTS]->(Claim)
  (Post)-[:USES]->(Source)
  (Topic)-[r:RELATED_TO]->(Topic) dove r.type DEVE ESSERE 'PREREQUISITO', 'CONTRASTO', 'ESTENSIONE', 'SIMILARE', 'SOTTO_CATEGORIA' o 'APPLICAZIONE'.
    
    ESEMPI DI RIFERIMENTO:
Utente: Trova i post del 2026 sul topic 'Docker' e restituisci claim e fonti (Source) associate.
Cypher: MATCH (p:Post)-[:COVERS]->(t:Topic) MATCH (p)-[:EXTRACTS]->(c:Claim) MATCH (p)-[:USES]->(s:Source) WHERE t.name =~ '(?i).*docker.*' AND p.date STARTS WITH '2026' RETURN p.title, c.text, s.name

Utente: Catena multi-hop in cui A è SIMILARE a B e B è un'ESTENSIONE di C, con motivi.
Cypher: MATCH (a:Topic)-[r1:RELATED_TO {type: 'SIMILARE'}]->(b:Topic)-[r2:RELATED_TO {type: 'ESTENSIONE'}]->(c:Topic) RETURN a.name, b.name, c.name, r1.reason, r2.reason

Utente: Conta quante fonti (Source) sono citate nei post che coprono il topic 'AI'.
Cypher: MATCH (p:Post)-[:COVERS]->(t:Topic) MATCH (p)-[:USES]->(s:Source) WHERE t.name =~ '(?i).*ai.*' WITH p, count(s) AS total_sources RETURN p.title, total_sources
    Richiesta: """

    prompt_text = """Genera SOLO la query Neo4j Cypher. Non aggiungere spiegazioni o formattazione markdown.
    REGOLE RIGIDE:
    1. Restituisci in RETURN SOLO le variabili esplicitamente richieste.
    2. Le date ('date') appartengono ESCLUSIVAMENTE al nodo Post (p.date). I Claim non hanno date.
    3. Mantieni i percorsi ottimali e diretti.

    SCHEMA CONSENTITO:
    - Nodi: Post {title, date}, Topic {name}, Claim {text}, Source {name}, Documentation {text}
    - Relazioni Post: (Post)-[:COVERS]->(Topic), (Post)-[:USES]->(Documentation), (Post)-[:EXTRACTS]->(Claim)
    - Relazioni Topic: (Topic)-[r:RELATED_TO]->(Topic) dove r.type DEVE ESSERE 'PREREQUISITO', 'CONTRASTO', 'ESTENSIONE', 'SIMILARE', 'SOTTO_CATEGORIA' o 'APPLICAZIONE'.

    ESEMPI DI RIFERIMENTO:
    Utente: Seleziona i post di aprile 2026 che coprono il topic 'Docker'.
    Cypher: MATCH (p:Post)-[:COVERS]->(t:Topic) WHERE p.date STARTS WITH '2026-04' AND t.name =~ '(?i).*docker.*' RETURN p.title
    Utente: Topic simili a 'RAG' senza contrasti.
    Cypher: MATCH (t1:Topic)-[:RELATED_TO {type: 'SIMILARE'}]->(t2:Topic) WHERE t2.name =~ '(?i).*rag.*' AND NOT (t1)-[:RELATED_TO {type: 'CONTRASTO'}]-(:Topic) RETURN t1.name

    Richiesta: """

    #full_prompt = prompt_text + query_testuale + "\nCypher: "

    full_prompt = prompt_text + query_testuale + "\nCypher: "

    try:
        response = requests.post(
            API_URL,
            json={"prompt": full_prompt},
            timeout=45
        )
        response.raise_for_status()
        data = response.json()
        return data.get("cypher_query", "").strip()

    except requests.exceptions.RequestException as e:
        return f"ERROR_API: {e}"


def test_query_on_neo4j(driver, cypher_query):
    """Esegue la query su Neo4j in READ mode. Restituisce: (is_valid, error_msg, is_write_attempt)"""
    try:
        driver.execute_query(
            cypher_query,
            routing_=RoutingControl.READ
        )
        return True, None, False
    except Neo4jError as e:
        # Catturiamo il tentativo di scrittura (MERGE/CREATE/SET) in modalità READ
        if e.code == "Neo.ClientError.Statement.AccessMode":
            return False, "Tentativo di scrittura bloccato.", True
        return False, f"Neo4jError [{e.code}]: {e.message}", False
    except Exception as e:
        return False, f"GenericError: {str(e)}", False


import re


def robust_cleaner(query):
    # 1. Rimuovi la funzione deprecata id()
    query = re.sub(r"id\(\w+\)\s*>\s*\d+", "elementId(n) > 0", query, flags=re.IGNORECASE)

    # 2. Correggi il pattern EXISTS mancante del MATCH
    # Trasforma: NOT EXISTS { (n)-[:REL]->(m) } -> NOT EXISTS { MATCH (n)-[:REL]->(m) }
    query = re.sub(r"NOT EXISTS\s*\{\s*(?!(?:MATCH|OPTIONAL|WHERE))", "NOT EXISTS { MATCH ", query, flags=re.IGNORECASE)
    # Chiudi la graffa correttamente se abbiamo aggiunto il MATCH
    # (Questa è una semplificazione, nei casi complessi controlla bene)

    # 3. Risolvi il doppio WHERE combinandoli con AND
    # Cerca un WHERE seguito da altro testo e un secondo WHERE
    if query.lower().count("where") > 1:
        parts = re.split(r'\bWHERE\b', query, flags=re.IGNORECASE)
        # Unisci la prima parte con le successive usando AND
        query = parts[0] + " WHERE " + " AND ".join(parts[1:])

    # 4. Rimuovi il GROUP BY (non supportato) e forzalo in un WITH
    # (Questo è un fix rudimentale, funziona bene per query semplici)
    if "GROUP BY" in query.upper():
        query = query.replace("GROUP BY", ",")
        # Assicura che ci sia un WITH prima
        if "WITH" not in query.upper():
            query = query.replace("RETURN", "WITH")

            # 5. Correggi le graffe doppie che il modello ha iniziato a scrivere
    query = query.replace("}}", "}").replace("]]", "]")

    # 6. Mappa le relazioni "USE" errate in "USES"
    query = query.replace("[:USE]", "[:USES]")
    query = query.replace("[:USE_SOURCE]", "[:USES]")

    return query

TEST_SOLO_ULTIMI_11 = False

def main():
    # 1. Caricamento dati di test
    try:
        test_items = load_test_dataset(DATASET_PATH)

        # Gestione switch modalità
        if TEST_SOLO_ULTIMI_11 and len(test_items) > 11:
            print("🚀 MODALITÀ: Esecuzione limitata agli ultimi 11 elementi.")
            test_items = test_items[-11:]
        else:
            print("🚀 MODALITÀ: Esecuzione su INTERO dataset.")

    except Exception as e:
        print(f"❌ Impossibile caricare il dataset: {e}")
        return

    num_total_items = len(test_items)
    # ... resto del codice rimane identico ...

    # 2. Connessione a Neo4j
    print("🔌 Connessione a Neo4j (Cloud)...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("✅ Connessione a Neo4j stabilita.")
    except Exception as e:
        print(f"❌ Errore di connessione a Neo4j: {e}")
        return

    # Inizializzazione di tutti i contatori
    successi = 0  # Sintassi corretta su Neo4j
    fallimenti = 0  # Errore di sintassi o connessione
    match_esatti = 0  # Stringa identica all'output atteso
    ignorati = 0  # Query MERGE/CREATE saltate

    print(f"\n🚀 Avvio del test su {num_total_items} campioni usando l'infrastruttura di progetto...\n")
    start_total_time = time.time()

    for idx, item in enumerate(test_items, 1):
        prompt_input = item.get("input")
        expected_output = item.get("output", "")

        print(f"{'=' * 80}")
        print(f"📝 TEST {idx}/{num_total_items}")
        print(f"📥 Input Utente: '{prompt_input}'")

        # Generazione tramite il server remoto
        start_gen = time.time()
        generated_cypher = generate_cypher(prompt_input)
        gen_time = time.time() - start_gen

        if "ERROR_API" in generated_cypher:
            print(f"❌ Errore di comunicazione con il server Flask: {generated_cypher}")
            fallimenti += 1
            continue

        print(f"🤖 Query Generata ({gen_time:.2f}s):\n   {generated_cypher}")
        print(f"🎯 Query Attesa:\n   {expected_output}")

        # 1. Controllo Exact Match
        is_exact = (generated_cypher.strip() == expected_output.strip())
        if is_exact:
            match_esatti += 1

        generated_cypher = robust_cleaner(generated_cypher)
        # 2. Controllo Sintattico Cloud e AccessMode
        is_valid, error_msg, is_write = test_query_on_neo4j(driver, generated_cypher)

        # 3. Stampe e attribuzione dei risultati
        if is_write:
            print("⚪ RISULTATO: IGNORATO (Query di Scrittura / Modifica dati)")
            ignorati += 1
        elif is_valid:
            if is_exact:
                print("🟢 RISULTATO: SINTASSI VALIDA | 🏆 MATCH ESATTO")
            else:
                print("🟡 RISULTATO: SINTASSI VALIDA | ❌ MATCH NON ESATTO")
            successi += 1
        else:
            print("🔴 RISULTATO: FALLITO (Errore di Sintassi)")
            print(f"⚠️ Dettaglio Errore:\n   {error_msg}")
            fallimenti += 1

    # ==========================================
    # REPORT FINALE
    # ==========================================
    total_time = time.time() - start_total_time
    driver.close()

    # Sottraiamo le query ignorate dal totale per avere una metrica reale
    query_valutate = num_total_items - ignorati

    accuracy_sintattica = (successi / query_valutate) * 100 if query_valutate > 0 else 0
    accuracy_esatta = (match_esatti / query_valutate) * 100 if query_valutate > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"📊 REPORT FINALE DI VALIDAZIONE")
    print(f"{'=' * 80}")
    print(f"⏱️ Tempo totale impiegato: {total_time:.2f} secondi")
    print(f"📦 Dataset totale: {num_total_items} query")
    print(f"⚪ Query di scrittura ignorate: {ignorati}")
    print(f"🔍 Query in sola lettura valutate: {query_valutate}")
    print("-" * 80)
    print(f"✅ Query Sintatticamente Corrette: {successi}/{query_valutate} ({accuracy_sintattica:.2f}%)")
    print(f"🏆 Query Esattamente Identiche:    {match_esatti}/{query_valutate} ({accuracy_esatta:.2f}%)")
    print(f"❌ Query Fallite/Errate:           {fallimenti}/{query_valutate}")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()