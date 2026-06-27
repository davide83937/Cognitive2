from datetime import datetime


from fine_tuned_model import generate_cypher
from function_tool import get_latest_scheduled_date_from_db, get_next_fixed_publish_date, \
    getRetriever, tavily_search_tool
from query_neo4j import get_post_count_by_date_query, get_all_topics_query
from langchain_core.tools import tool
from test_neo4j import graph


@tool
def write_an_article(about: str, author: str, content: str):
    """Write an article about the topic given by user, author is IA"""
    return f"My article about: {about}. \n{content} \n Written by {author}"


@tool
def schedule_manager_tool(data_target: str = None) -> str:
    """
    Strumento UNIFICATO per la gestione del palinsesto e delle date degli articoli.
    - Se l'utente propone una data, passa 'data_target' (formato YYYY-MM-DD) per verificare se è disponibile.
    - Se l'utente chiede la prima data libera (o non specifica nulla), NON passare 'data_target' (lascialo vuoto o null).
    Regole del blog: pubblicazione solo di Lunedì, Mercoledì, Venerdì e Domenica (massimo 3 post al giorno).
    """
    if not graph:
        return "Errore: Database Neo4j non connesso."

    giorni_pubblicazione = [0, 2, 4, 6]
    giorni_nomi = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]

    # ==========================================
    # CASO 1: VERIFICA DI UNA DATA SPECIFICA
    # ==========================================
    if data_target and data_target.lower() != 'null':
        try:
            target_date_obj = datetime.strptime(data_target, "%Y-%m-%d").date()
        except ValueError:
            return "Errore: Formato data non valido, usa YYYY-MM-DD."

        giorno_settimana = target_date_obj.weekday()
        if giorno_settimana not in giorni_pubblicazione:
            return (f"Rifiutato: La data {data_target} è un {giorni_nomi[giorno_settimana]}. "
                    f"Il blog pubblica solo di Lun, Mer, Ven e Dom. Prova a cercare la prima data disponibile.")

        # L'IA usa il fine-tuning per estrarre i post di quel giorno
        nl_instruction = f"Restituisci i titoli dei post che hanno la data uguale a '{data_target}'."
        cypher_query = generate_cypher(nl_instruction)
        print(f"🔧 [DEBUG TEXT-TO-CYPHER] Check Data Specifica:\n{cypher_query}")

        try:
            risultati = graph.query(cypher_query)
            current_count = len(risultati) if risultati else 0

            if current_count < 3:
                return (f"La data {data_target} ({giorni_nomi[giorno_settimana]}) è DISPONIBILE. "
                        f"Ci sono {current_count} articoli pianificati.")
            else:
                return f"La data {data_target} è OCCUPATA. Limite massimo di 3 articoli raggiunto."
        except Exception as e:
            return f"Errore Cypher: {e}"

    # ==========================================
    # CASO 2: RICERCA DELLA PRIMA DATA LIBERA
    # ==========================================
    else:
        # Recuperiamo da dove partire
        base_date_str = get_latest_scheduled_date_from_db()
        giorno_corrente = get_next_fixed_publish_date(base_date_str)

        # L'IA usa il fine-tuning per estrarre il calendario completo
        nl_instruction = "Restituisci la data di tutti i post presenti nel database."
        cypher_query = generate_cypher(nl_instruction)
        print(f"🔧 [DEBUG TEXT-TO-CYPHER] Analisi Calendario Globale:\n{cypher_query}")

        # Costruiamo il calendario in Python
        date_occupate = {}
        try:
            risultati = graph.query(cypher_query)
            for r in risultati:
                valori = list(r.values())
                if valori and valori[0]:
                    data_post = str(valori[0])
                    date_occupate[data_post] = date_occupate.get(data_post, 0) + 1
        except Exception as e:
            return f"Errore Cypher: {e}"

        # Loop ibrido super-veloce per trovare il giorno libero
        for _ in range(50):
            data_str = giorno_corrente.strftime("%Y-%m-%d")
            articoli_presenti = date_occupate.get(data_str, 0)

            if articoli_presenti < 3:
                return f"La prima data disponibile in palinsesto nel Knowledge Graph è: {data_str}"

            giorno_corrente = get_next_fixed_publish_date(data_str)

        return "Non ci sono date disponibili nel palinsesto."





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
    # Usiamo il vocabolario dello schema: Post -> EXTRACTS -> Claim, Post -> COVERS -> Topic
    nl_instruction = f"Restituisci il testo dei claim estratti dai post che coprono il topic '{topic_name}'."

    cypher_query = generate_cypher(nl_instruction)
    print(f"🔧 [DEBUG T2C] Query:\n{cypher_query}")

    try:
        risultati = graph.query(cypher_query)
        # Estrai il primo valore restituito dinamicamente
        lista_claims = [str(list(res.values())[0]) for res in risultati]
        return "\n- ".join(lista_claims) if lista_claims else "Nessun claim trovato."
    except Exception as e:
        return f"Errore Cypher: {e}"

#@tool
#def get_topic_claims(topic_name: str) -> str:
    """
    Usa questo tool durante la stesura dell'articolo per recuperare le affermazioni chiave (Claims)
    fatte in passato su un determinato topic (topic_name), così da mantenere coerenza.
    """
    """from test_neo4j import graph  # Assicurati di avere l'import
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
"""

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
    Seleziona e restituisci SOLO i risultati che provengono da fonti affidabili.
    
    CRITERI DI AFFIDABILITA' (Devono rispettare almeno uno):
    1. Provenienza da domini accademici o governativi (.edu, .gov).
    2. Pubblicazioni su portali scientifici riconosciuti (es. arxiv.org, ieee.org, nature.com).
    3. Articoli che presentano chiaramente l'autore, la sua affiliazione e citano studi o dataset esterni linkabili.
    
    CRITERI DI INAFFIDABILITA' (Scarta la fonte se presenta uno di questi):
    1. È un forum, un sito di Q&A (Reddit, Quora) o un'enciclopedia libera (Wikipedia).
    2. Usa un linguaggio sensazionalistico o titoli clickbait.
    3. È un blog personale senza credenziali o un sito con evidente intento promozionale/commerciale.
    4. Fa affermazioni tecniche o scientifiche senza citare uno studio o una fonte primaria.

    Rispondi fornendo un breve riepilogo delle fonti salvate e il loro contenuto utile. E spiega brevemente quali hai 
    scartato e perché.
    Se nessuna fonte è buona, scrivi "Nessuna fonte affidabile trovata".
    """

    risposta_filtrata = llm_giudice.invoke([{"role": "user", "content": prompt_giudice}])

    # AGGIUNGIAMO QUESTO PRINT PER VEDERE IL RAGIONAMENTO DEL GIUDICE
    print(f"\n⚖️ [DEBUG GIUDICE FONTI] Verdetto sulle fonti trovate:\n{risposta_filtrata.content}\n")

    return risposta_filtrata.content



@tool
def intelligent_topic_matcher(new_topic: str) -> str:
    """
    Usa questo tool per scoprire SE e COME un nuovo argomento è già stato trattato nel Knowledge Graph.
    Fornisce la corrispondenza semantica esatta del nome del Topic nel database.
    """
    from Models import get_llm
    if not graph:
        return "Errore: Database Neo4j non connesso."

    # 1. GENERAZIONE DELLA QUERY CON IL MODELLO FINE-TUNED
    # Chiediamo i topic e i claim seguendo lo schema del training: Topic <-[:COVERS]- Post -[:EXTRACTS]-> Claim
    nl_instruction = "Restituisci il nome di tutti i topic presenti e i testi dei claim estratti dai post che coprono ciascun topic."

    cypher_query = generate_cypher(nl_instruction)
    print(f"🔧 [DEBUG TEXT-TO-CYPHER] Query Topic Matcher:\n{cypher_query}")

    # 2. ESECUZIONE ED ELABORAZIONE DEI RISULTATI IN PYTHON
    try:
        records = graph.query(cypher_query)
        if not records:
            return f"Il database è vuoto. '{new_topic}' è un argomento 100% nuovo."

        # Usiamo un dizionario per raggruppare i claim sotto lo stesso topic (emulando il collect[] di Cypher)
        topic_dict = {}
        for r in records:
            valori = list(r.values())
            if len(valori) >= 1:
                nome_topic = str(valori[0])
                # Controlliamo se la query ha restituito anche un claim valido nella seconda colonna
                claim = str(valori[1]) if len(valori) > 1 and valori[1] is not None else None

                if nome_topic not in topic_dict:
                    topic_dict[nome_topic] = []

                # Salviamo al massimo 2 claim di esempio per non saturare il prompt del Giudice
                if claim and claim != "None" and len(topic_dict[nome_topic]) < 2:
                    topic_dict[nome_topic].append(claim)

        # Costruiamo il contesto testuale per l'LLM
        lista_esistenti = []
        for nome, claims in topic_dict.items():
            lista_esistenti.append(f"- Topic: '{nome}' (Esempi trattati: {claims})")

        contesto_db = "\n".join(lista_esistenti)

    except Exception as e:
        return f"Errore durante l'interrogazione di Neo4j: {e}"

    # 3. IL GIUDICE SEMANTICO (Rimane invariato, usa un LLM standard come GPT-4o)
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

    # 4. RISPOSTA FINALE ALL'AGENTE LANGGRAPH
    if risposta_llm == "NESSUNO":
        return f"Nessuna corrispondenza semantica trovata. L'argomento '{new_topic}' è nuovo."
    else:
        return f"Trovata corrispondenza semantica! Nel database l'argomento è salvato ESATTAMENTE con il nome: '{risposta_llm}'. Usa questo nome per interrogare i claims."


#@tool
#def intelligent_topic_matcher(new_topic: str) -> str:
    """
    Usa questo tool per scoprire SE e COME un nuovo argomento è già stato trattato nel Knowledge Graph.
    Fornisce la corrispondenza semantica esatta del nome del Topic nel database.
    """
    """from Models import get_llm
    if not graph:
        return "Errore: Database Neo4j non connesso."
        """

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


#@tool
#def get_enhanced_topic_context(topic_name: str) -> str:
    """
    Usa questo tool durante la stesura dell'articolo per recuperare il contesto profondo
    dal Knowledge Graph: estrae i claim storici sul topic e la mappa di relazioni
    con altri argomenti (es. prerequisiti, sotto-categorie) con le relative motivazioni.
    """
    """if not graph:
        return "Errore: Database Neo4j non connesso."

    from query_neo4j import get_enhanced_topic_context_query
    query = get_enhanced_topic_context_query()

    print(f"\n🧠 [DEBUG KG] Estrazione contesto arricchito per il topic: '{topic_name}'")

    try:
        risultati = graph.query(query, params={"topic": topic_name})
        if not risultati or not risultati[0].get("topic_name"):
            return f"Nessuna informazione storica o relazione trovata nel KG per il topic '{topic_name}'."

        record = risultati[0]
        claims = record.get("direct_claims", [])
        relations = record.get("relations", [])

        output = f"--- CONTESTO KNOWLEDGE GRAPH PER: '{topic_name}' ---\n"

        # Formattazione dei Claim storici
        if claims:
            output += "\n📌 Concetti e claim già trattati in passato:\n- " + "\n- ".join(claims) + "\n"
        else:
            output += "\n📌 Nessun claim registrato in precedenza su questo specifico topic.\n"

        # Formattazione della struttura del Grafo
        valid_relations = [r for r in relations if r.get('target')]
        if valid_relations:
            output += "\n🔗 Mappa delle Relazioni Semantiche nel Blog:\n"
            for rel in valid_relations:
                output += f"- Connesso a '{rel['target']}' | Tipo: [{rel['type']}] | Motivo: {rel['reason']}\n"
        else:
            output += "\n🔗 Nessuna connessione strutturale con altri macro-topic nel grafo.\n"

        return output
    except Exception as e:
        return f"Errore durante l'interrogazione avanzata del KG: {e}"
    """

@tool
def get_enhanced_topic_context(topic_name: str) -> str:
    """
    Recupera il contesto profondo: claim storici e mappa di relazioni semantiche.
    """
    if not graph:
        return "Errore: Database Neo4j non connesso."

    # Chiediamo esplicitamente le variabili per sfruttare la RELAZIONE TOPIC del tuo schema
    nl_instruction = f"Restituisci il nome del topic correlato e il tipo di relazione per il topic '{topic_name}'."

    cypher_query_rel = generate_cypher(nl_instruction)
    print(f"🔧 [DEBUG TEXT-TO-CYPHER] Query Relazioni:\n{cypher_query_rel}")

    # Per i claim usiamo un'altra query diretta per semplicità e precisione dell'LLM
    nl_instruction_claims = f"Restituisci il testo dei claim estratti dai post che coprono il topic '{topic_name}'."
    cypher_query_claims = generate_cypher(nl_instruction_claims)

    try:
        # Eseguiamo le interrogazioni (puoi anche unirle se il tuo LLM sa gestire query complesse con OPTIONAL MATCH)
        risultati_rel = graph.query(cypher_query_rel)
        risultati_claims = graph.query(cypher_query_claims)

        output = f"--- CONTESTO KNOWLEDGE GRAPH PER: '{topic_name}' ---\n"

        if risultati_claims:
            claims = set([str(list(r.values())[0]) for r in risultati_claims])
            output += "\n📌 Concetti già trattati in passato:\n- " + "\n- ".join(claims) + "\n"

        if risultati_rel:
            output += "\n🔗 Mappa delle Relazioni:\n"
            for rel in risultati_rel:
                # Estrazione flessibile dei valori restituiti
                valori = list(rel.values())
                if len(valori) >= 2:
                    output += f"- Connesso a '{valori[0]}' | Tipo: [{valori[1]}]\n"

        return output
    except Exception as e:
        return f"Errore Cypher: {e}"