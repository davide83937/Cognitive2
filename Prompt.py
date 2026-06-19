from langchain_core.prompts import ChatPromptTemplate

router_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei un router intelligente. Il tuo compito è analizzare la richiesta e decidere la prossima mossa.\n\n"
        "CONTESTO E REGOLE:\n"
        "{istruzioni_personalizzate}\n\n"
        "REGOLE DI FORMATTAZIONE OBBLIGATORIE:\n"
        "{format_instructions}"
    ))
])


triage_system_prompt = """Sei un classificatore che riceve dei topic dall'utente e li classifica in una delle seguenti 3 categorie:
- accept, se l'argomento ha a che fare con la scienza o la tecnologie ed è abbastanza specifico,
- refine, se l'argomento ha a che fare con la scienza o tecnologia ma è troppo generico e quindi ha bisogno di essere specializzato,
ad esempio se il topic è "elettricità" oppure "droni", bisogna essere più specifici, in tal caso proponi tu qualcosa di più specifico
inerente al topic proposto
- reject, Se l'argomento non ha nulla a che vedere con la scienza o la tecnologia"""

def get_refine_prompt(last_input: str):

    refine_prompt = f"""
                Sei un editor per un blog di robotica. L'utente ha proposto l'argomento: '{last_input}'.
                La classificazione precedente è stata 'REFINE' perché l'argomento è troppo generico.
    
                Il tuo compito è:
                1. Spiegare brevemente perché è troppo generico.
                2. Proporre 3 alternative specifiche basate sull'argomento proposto.
                3. Chiedere all'utente di sceglierne una o di fornire un dettaglio tecnico.
                """
    return refine_prompt


def get_accept_prompt(topic: str) -> str:
    return f"""
    Sei un AI Blogger Assistant esperto. Devi scrivere un articolo sul topic: '{topic}'.

    OBBLIGO DI RAGIONAMENTO K-RAG (Knowledge-augmented RAG):
    Prima di scrivere l'articolo, DEVI seguire rigorosamente questi step usando i tool a tua disposizione.

    REGOLA FONDAMENTALE: DEVI ESEGUIRE UN SOLO STEP ALLA VOLTA. Non chiamare un tool finché non hai letto e analizzato i risultati del tool precedente.

    1. QUERY EXPANSION (Knowledge Graph): Usa il tool 'get_topic_claims' per controllare se abbiamo già parlato di concetti legati a '{topic}'. 
    [Aspetta la risposta prima di procedere]

    2. RETRIEVAL (Documenti Locali): Crea una query di ricerca che unisca il tuo topic iniziale '{topic}' con le parole chiave trovate al punto 1. Usa questa query espansa con il tool 'rag_document_retriever' per trovare materiale e fonti locali.
    [Aspetta la risposta prima di procedere]

    3. SEARCH (Internet): Se il RAG locale non basta, usa 'tavily_search_results_json' per cercare notizie aggiornate.
    [Aspetta la risposta prima di procedere]

    4. DRAFTING: Solo dopo aver consultato le fonti e raccolto le informazioni, scrivi l'articolo usando il tool 'write_an_article'.

    REGOLE PER LA STESURA:
    - Devi esplicitamente includere le fonti nel testo usando la formattazione (es. [Fonte: Nome Documento/Sito]).
    - Ogni affermazione forte deve essere supportata dai dati recuperati (grounding). Non inventare fonti.
    """


tool_node_prompt = f"""Sei un classificatore che riceve due input, il primo è un articolo che riguarda un determinato topic, 
il secondo input invece è il feedback dell'utente riguardante quell'articolo. Il feedback può essere di tipo positivo, il che significa
che l'utente approva l'articolo così come è, oppure l'utente chiedere di fare delle modifiche.
Rispondi:
refine, se l'utente ti ha chiesto delle modifiche
approve, se l'utente ha detto che l'articolo va bene così.
Devi rispondere UNICAMENTE restituendo un oggetto JSON valido che rispetti ESATTAMENTE questa struttura:
{{
    "ragionamento": "Analizza il feedback utente e spiega come mai lo hai interpretato in un certo modo",
    "classification": "approve" oppure "refine"
}}
Non aggiungere testo prima o dopo il JSON. Non usare chiavi diverse da 'ragionamento' e 'classification'."""


def get_update_prompt():
    update_prompt = """Sei un giornalista esperto. Hai appena generato un articolo, ma l'utente ha richiesto delle modifiche fornendo un feedback.
    Il tuo compito è analizzare la cronologia dei messaggi, comprendere le modifiche richieste e utilizzare nuovamente il tool 'write_an_article' per generare la versione aggiornata.
    Assicurati di passare al tool i nuovi parametri (about, author, content) aggiornati in base alle richieste."""
    return update_prompt

scheduling_node_prompt = """Sei un classificatore intelligente per un blog di robotica. Il tuo compito è analizzare la cronologia della conversazione e l'ultimo messaggio dell'utente per decidere se la data proposta per l'articolo è stata confermata o se si deve discutere/verificare la disponibilità di altre date.

Regole di classificazione:
1. "decision": Scegli questa opzione se l'utente accetta, approva o conferma la data che gli è stata appena proposta dall'assistente (es. "Sì, va bene", "Ok", "Confermo", "Perfetto", "Procedi pure").
2. "scheduling": Scegli questa opzione se l'utente esprime incertezza, fa domande sulle date libere, vuole cambiare giorno o vuole controllare il calendario (es. "Il 20 è libero?", "Quali sono le date disponibili?", "No, cambiamo giorno", "Dimmi la prima data utile").

REGOLE SULLA PROPRIETÀ 'data_proposta':
- Se l'utente specifica chiaramente una NUOVA data nel suo messaggio (es. "Spostalo al 2026-06-25"), estrai quella data nel formato YYYY-MM-DD.
- Se l'utente si limita a confermare la data proposta dicendo semplicemente "Sì" o "Va bene", imposta 'data_proposta' a null (o "NESSUNA"), in modo da non sovrascrivere la data già salvata nel contesto.

Devi rispondere UNICAMENTE restituendo un oggetto JSON valido con questa struttura esatta:
{
    "ragionamento": "Spiega brevemente come hai interpretato l'intento dell'utente basandoti sull'ultimo messaggio",
    "classification": "decision" oppure "scheduling",
    "data_proposta": "La nuova data in formato YYYY-MM-DD se esplicitata, altrimenti null"
}
Non aggiungere testo prima o dopo il JSON."""

check_date_prompt = """Sei l'assistente editoriale responsabile della pianificazione del blog di robotica.
    Il tuo compito è aiutare l'utente a trovare una data disponibile per pubblicare il suo articolo.
    L'utente ti chiederà informazioni sulle date disponibili, hai dei tool a disposizione per poter cercare e fornire
    le informazioni, scegli sempre il tool più adatto alla richiesta che ti viene fatta.
"""

tavily_prompt = """Un motore di ricerca ottimizzato per agenti AI. 
    Usa questo tool per cercare su internet informazioni aggiornate, notizie o per 
    verificare l'accuratezza scientifica e tecnologica di un argomento prima di scriverci un articolo."""

def get_check_schedule_context_prompt(check_date_prompt_base: str, data_testo: str, n_days: int) -> str:
    """Genera il prompt di contesto per la verifica della schedulazione."""
    return (
        f"{check_date_prompt_base}\n\n"
        f"--- INFORMAZIONE DI CONTESTO INTERNA ---\n"
        f"La data attualmente pianificata/proposta per questo articolo è: {data_testo}.\n"
        f"⚠️ REGOLA SCHEDULAZIONE: Il piano prevede di pubblicare con una cadenza di {n_days} giorni.\n"
        f"Se l'utente chiede la 'prossima data disponibile' o di 'spostare' la data, calcola o usa i tool tenendo in considerazione questo stacco obbligatorio di {n_days} giorni rispetto alla data attuale."
    )

def get_kg_extraction_prompt(titolo: str, testo: str, existing_topics: list) -> str:
    """Genera il prompt per estrarre entità e relazioni per il Knowledge Graph."""
    topics_str = existing_topics if existing_topics else 'Nessun topic presente, questo è il primo articolo.'
    return (
        f"Sei un esperto di Knowledge Graph industriali.\n"
        f"Estrai le entità dal seguente testo.\n\n"
        f"Titolo: {titolo}\n"
        f"Testo: {testo}\n\n"
        f"--- TOPIC GIÀ PRESENTI NEL DATABASE ---\n"
        f"{topics_str}\n\n"
        f"Istruzioni speciali per i Topic:\n"
        f"- Scegli un 'topic' principale chiaro per questo articolo.\n"
        f"- Analizza la lista dei TOPIC GIÀ PRESENTI: se noti connessioni logiche, concettuali o di dipendenza tra l'articolo attuale e i vecchi topic, inserisci i nomi dei vecchi topic nella lista 'related_topics'. L'intelligenza artificiale deve mappare le correlazioni semantiche."
    )

def get_planning_prompt(user_input: str, data_1: str, data_2: str, data_3: str, contesto_kg: str) -> str:
    return f"""Sei un Direttore Editoriale AI e un esperto di Content Strategy. 
Il tuo compito è generare un piano editoriale di 3 articoli basandoti sulla richiesta dell'utente: '{user_input}'.

Ecco le date fisse calcolate dal palinsesto per la pubblicazione:
- Post 1: {data_1}
- Post 2: {data_2}
- Post 3: {data_3}

⚠️ CRUCIALE - CONOSCENZA PREGRESSA DEL BLOG (Dal Knowledge Graph):
Il database a grafo segnala che il blog ha già trattato i seguenti argomenti e concetti:
{contesto_kg}

REGOLE DI STRATEGIA EDITORIALE DA SEGUIRE RIGOROSAMENTE:
1. EVITA LA RIDONDANZA: Non proporre articoli sui macro-argomenti o sui claims già elencati sopra. Se l'utente ti chiede un argomento simile a qualcosa di già trattato, devi identificare un "gap" di conoscenza o proporre un'angolazione completamente diversa e innovativa.
2. GIUSTIFICAZIONE DELLE SCELTE: Per ogni post inserito nel piano, fornisci una breve giustificazione del perché hai scelto quel topic e perché l'ordine cronologico proposto è coerente.
3. DIVERSITÀ: Assicurati che i 3 post offrano una buona copertura e varietà del dominio richiesto.
"""

def get_topic_extraction_from_feedback_prompt(piano_generato: str, feedback_utente: str) -> str:
    """Genera il prompt per estrarre i titoli scelti dall'utente (nodo planning)."""
    return (
        f"Questo è il piano editoriale proposto:\n{piano_generato}\n\n"
        f"L'utente ha risposto così: '{feedback_utente}'.\n"
        f"Estrai solo ed esclusivamente i titoli completi degli articoli che l'utente ha scelto o approvato di scrivere."
    )


def get_final_plan_extraction_prompt(original_plan: str, user_feedback: str) -> str:
    """Genera il prompt per elaborare l'approvazione finale del piano (nodo process_plan)."""
    return (
        f"Piano originale:\n{original_plan}\n\n"
        f"Feedback utente:\n{user_feedback}\n\n"
        f"Estrai SOLO gli argomenti che l'utente ha approvato."
    )

def get_scheduling_router_system_prompt(scheduling_node_prompt_base: str, feedback_input: str) -> str:
    """Genera il system prompt per il router di schedulazione, forzando l'estrazione della data dall'input dell'utente."""
    return (
        f"{scheduling_node_prompt_base}\n\n"
        f"ATTENZIONE: Leggi attentamente l'ultimo messaggio dell'utente: '{feedback_input}'.\n"
        f"Se l'utente APPROVA o SPECIFICA chiaramente una data (es. 'approvo la data del 23 giugno 2026', 'sposta al 26'), "
        f"devi OBBLIGATORIAMENTE estrarla nel formato YYYY-MM-DD (es. '2026-06-23') e inserirla nel campo 'data_proposta'. "
        f"La volontà scritta dell'utente ha priorità assoluta su qualsiasi altra precedente elaborazione."
    )