from datetime import datetime, timedelta
from test_neo4j import graph
from fine_tuned_model import generate_cypher
from function_tool import getRetriever, tavily_search_tool
from query_neo4j import get_all_topics_query
from langchain_core.tools import tool



@tool
def write_an_article(about: str, author: str, content: str):
    """Write an article about the topic given by user, author is IA"""
    return f"My article about: {about}. \n{content} \n Written by {author}"


@tool
def calendar_query_tool(natural_language_query: str) -> str:
    """
    Strumento generalizzato per interrogare il palinsesto del blog nel Knowledge Graph.
    Passa una richiesta in linguaggio naturale (es. "Quanti articoli ci sono il 2026-05-10?",
    "Qual è l'ultima data occupata?", "Dammi tutte le date dei post pianificati").
    Il tool tradurrà la richiesta in Cypher e restituirà i risultati grezzi.
    """
    if not graph:
        return "Errore: Database Neo4j non connesso."

    # 1. Deleghiamo al modello fine-tuned sulla RTX 3070 la creazione della query
    cypher_query = generate_cypher(natural_language_query)
    print(f"🔧 [DEBUG TEXT-TO-CYPHER] Query generata per il calendario:\n{cypher_query}")

    # 2. Esecuzione diretta su Neo4j
    try:
        risultati = graph.query(cypher_query)

        if not risultati:
            return "Nessun risultato trovato nel database. Il palinsesto potrebbe essere vuoto per questa ricerca."

        # 3. Formattazione pulita dei risultati da restituire all'agente
        # Trasformiamo la lista di dizionari in una stringa leggibile per l'LLM
        output = "Risultati estratti dal DB:\n"
        for row in risultati:
            output += f"- {row}\n"

        return output

    except Exception as e:
        return f"Errore durante l'esecuzione in Neo4j: {e}\nQuery generata: {cypher_query}"





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
    Usa questo tool per scoprire SE e QUALI argomenti sono già stati trattati nel Knowledge Graph.
    Fornisce la corrispondenza semantica dei Topic nel database.
    """
    from Models import get_llm
    from test_neo4j import graph

    if not graph:
        return "Errore: Database Neo4j non connesso."

    nl_instruction = (
        "Restituisci il nome di tutti i topic presenti e i testi dei claim estratti dai post che coprono ciascun topic. Limita la tua ricerca solo agli ultimi 30 giorni")
    cypher_query = generate_cypher(nl_instruction)

    try:
        records = graph.query(cypher_query)
        if not records:
            return f"Il database è vuoto. '{new_topic}' è un argomento 100% nuovo."

        topic_dict = {}
        for r in records:
            valori = list(r.values())
            if len(valori) >= 1:
                nome_topic = str(valori[0])
                claim = str(valori[1]) if len(valori) > 1 and valori[1] is not None else None

                if nome_topic not in topic_dict:
                    topic_dict[nome_topic] = []

                if claim and claim != "None" and len(topic_dict[nome_topic]) < 2:
                    topic_dict[nome_topic].append(claim)

        lista_esistenti = []
        for nome, claims in topic_dict.items():
            lista_esistenti.append(f"- Topic: '{nome}' (Esempi trattati: {claims})")

        contesto_db = "\n".join(lista_esistenti)

    except Exception as e:
        return f"Errore durante l'interrogazione di Neo4j: {e}"

    # --- 3. IL GIUDICE SEMANTICO AGGIORNATO PER MULTI-TOPIC ---
    llm_giudice = get_llm()
    prompt_giudice = f"""
    Sei un analista semantico. Il tuo compito è confrontare un NUOVO argomento con una lista di argomenti GIA' ESISTENTI nel database.

    NUOVO ARGOMENTO PROPOSTO: "{new_topic}"

    ARGOMENTI ESISTENTI NEL DATABASE:
    {contesto_db}

    DOMANDA: Il nuovo argomento proposto è concettualmente uguale, affine, o una sotto-categoria di uno o più argomenti esistenti?
    Identifica TUTTI i topic esistenti che hanno una forte correlazione semantica con il nuovo argomento.

    REGOLE DI RISPOSTA (IMPORTANTISSIMO):
    - Se trovi uno o più topic corrispondenti, rispondi SOLO con i NOMI ESATTI dei Topic esistenti separati da una virgola (es. "Droni, Sistemi Autonomi"). Non aggiungere altre parole, spiegazioni o introduzioni.
    - Se NON c'è nessuna corrispondenza semantica con nessun topic, rispondi ESATTAMENTE con la parola: "NESSUNO".
    """

    risposta_llm = llm_giudice.invoke([{"role": "user", "content": prompt_giudice}]).content.strip()

    # --- 4. RISPOSTA FINALE MULTI-TOPIC ALL'AGENTE ---
    if risposta_llm == "NESSUNO":
        return f"Nessuna corrispondenza semantica trovata. L'argomento '{new_topic}' è nuovo."
    else:
        # Restituiamo l'elenco dei topic trovati sotto forma di stringa strutturata
        return f"CORRISPONDENZA_TROVATA: {risposta_llm}"



@tool
def get_flexible_schedule_dates(start_date: str, end_date: str, limit: int = 10) -> str:
    """
    Trova date disponibili per la pubblicazione (Lun, Mer, Ven, Dom con < 3 post)
    all'interno di un range di tempo.
    Usa 'start_date' ed 'end_date' nel formato YYYY-MM-DD.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return "Errore: le date devono essere nel formato YYYY-MM-DD."

    # 1. Troviamo TUTTE le date candidate nel range
    giorni_validi = [0, 2, 4, 6]
    tutte_le_date_candidate = []

    corrente = start
    while corrente <= end:
        if corrente.weekday() in giorni_validi:
            tutte_le_date_candidate.append(corrente.strftime("%Y-%m-%d"))
        corrente += timedelta(days=1)

    if not tutte_le_date_candidate:
        return f"Nel periodo indicato ({start_date} al {end_date}) non ci sono giorni di palinsesto validi."

    date_disponibili_finali = []

    # 2. BATCHING: Dividiamo le date in piccoli lotti da 5 per evitare limiti di token
    batch_size = 5
    print(f"🤖 [TOOL NLP] Trovate {len(tutte_le_date_candidate)} date candidate. Inizio elaborazione a lotti...")

    for i in range(0, len(tutte_le_date_candidate), batch_size):
        lotto_corrente = tutte_le_date_candidate[i:i + batch_size]

        # Creiamo la richiesta NLP solo per questo piccolo gruppo
        date_str_nlp = ", ".join([f"'{d}'" for d in lotto_corrente])
        prompt_nlp_per_cypher = f"Restituisci la data (p.date) e il conteggio dei post raggruppati per data, solo per i post dove la data è una di queste: {date_str_nlp}."

        cypher_query = generate_cypher(prompt_nlp_per_cypher)

        conteggio_db = {}
        if graph:
            try:
                risultati = graph.query(cypher_query)
                if risultati:
                    for res in risultati:
                        # Estraiamo la data (solitamente p.date o date)
                        data_db = res.get("p.date") or res.get("date")

                        # ---> INIZIO FIX: Estrazione Dinamica del Conteggio <---
                        # Cerchiamo dinamicamente tra tutti i valori della riga restituita dal DB
                        # Il conteggio sarà sicuramente un numero intero (int)
                        count = 1
                        for valore in res.values():
                            if isinstance(valore, int):
                                count = valore
                                break
                        # ---> FINE FIX <---

                        if data_db:
                            conteggio_db[data_db] = count
            except Exception as e:
                print(f"⚠️ Errore Cypher sul lotto corrente (salto): {e}")

        # 3. Controllo disponibilità per il lotto analizzato
        for data_cand in lotto_corrente:
            occupazione = conteggio_db.get(data_cand, 0)
            if occupazione < 3:
                date_disponibili_finali.append(data_cand)

                # Se abbiamo già raggiunto il limite di date richieste dall'LLM, usciamo da tutto!
                if len(date_disponibili_finali) == limit:
                    break

                    # Interrompiamo anche il ciclo principale dei lotti se abbiamo raggiunto il limite
        if len(date_disponibili_finali) == limit:
            break

    if not date_disponibili_finali:
        return f"Attenzione: Tutti i giorni di palinsesto tra il {start_date} e il {end_date} hanno già 3 post programmati."

    return f"Date disponibili trovate con successo (meno di 3 post): {', '.join(date_disponibili_finali)}."



@tool
def get_enhanced_topic_context(topic_name: str) -> str:
    """
    Recupera il contesto profondo: claim storici e mappa di relazioni semantiche.
    """
    if not graph:
        return "Errore: Database Neo4j non connesso."

    # 1. NLP per gli archi (chiediamo esplicitamente tipo e motivo)
    nl_instruction_rel = f"Restituisci il nome del topic correlato, il tipo di relazione e il motivo della relazione per il topic '{topic_name}'."
    cypher_query_rel = generate_cypher(nl_instruction_rel)

    # 2. NLP per i claim (inalterato)
    nl_instruction_claims = f"Restituisci il testo dei claim estratti dai post che coprono il topic '{topic_name}'."
    cypher_query_claims = generate_cypher(nl_instruction_claims)

    try:
        risultati_rel = graph.query(cypher_query_rel)
        risultati_claims = graph.query(cypher_query_claims)

        output = f"--- CONTESTO KNOWLEDGE GRAPH PER: '{topic_name}' ---\n"

        # --- GESTIONE E DEBUG DEI CLAIM ---
        if risultati_claims:
            claims = set([str(list(r.values())[0]) for r in risultati_claims])
            print("\n" + "=" * 50)
            print(f"🔍 [DEBUG CLAIM] Trovati {len(claims)} claim per '{topic_name}':")
            for c in claims:
                print(f"  - {c}")

            output += "\n📌 Concetti già trattati in passato:\n- " + "\n- ".join(claims) + "\n"

        # --- GESTIONE E DEBUG DELLE RELAZIONI (ARCHI) ---
        if risultati_rel:
            print("\n" + "=" * 50)
            print(f"🔗 [DEBUG ARCHI E RELAZIONI] Nodi collegati a '{topic_name}':")

            output += "\n🔗 Mappa delle Relazioni:\n"
            for rel in risultati_rel:
                # Stampiamo il dizionario grezzo restituito da Neo4j per ispezione
                print(f"  -> RAW RECORD NEO4J: {rel}")

                valori = list(rel.values())

                # Se l'LLM ha estratto correttamente Nome, Tipo e Motivo (3 campi)
                if len(valori) >= 3:
                    nome_target = valori[0]
                    tipo_rel = valori[1]
                    motivo_rel = valori[2]

                    print(f"  -> ESTRATTO: Target='{nome_target}', Type='{tipo_rel}', Reason='{motivo_rel}'")
                    output += f"- Connesso a '{nome_target}' | Tipo: [{tipo_rel}] | Motivo: {motivo_rel}\n"

                # Fallback se ne estrae solo 2 (Nome e Tipo)
                elif len(valori) == 2:
                    print(f"  -> ESTRATTO (No Reason): Target='{valori[0]}', Type='{valori[1]}'")
                    output += f"- Connesso a '{valori[0]}' | Tipo: [{valori[1]}]\n"

            print("=" * 50 + "\n")

        return output
    except Exception as e:
        return f"Errore Cypher: {e}"