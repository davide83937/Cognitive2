import uuid
from langsmith import Client
from langsmith.evaluation import evaluate
from langgraph.types import Command
from langchain_core.messages import HumanMessage

# Assicurati che i percorsi di importazione siano corretti per il tuo progetto
from Test.test_models import call_llm_judge
from Main import app

# ==========================================
# 1. DEFINIZIONE DEL DATASET
# ==========================================
dataset_topics = [
    # Robotica e Software (Originali e varianti)
    {"topic": "Le ultime novità sui bracci robotici open-source nel 2026"},
    {"topic": "Come integrare ROS2 Humble con un database a grafo Neo4j"},
    {"topic": "Implementazione di algoritmi di Reinforcement Learning per la navigazione autonoma dei droni"},

    # Intelligenza Artificiale
    {"topic": "Il ruolo dell'Intelligenza Artificiale Generativa nella scoperta di nuovi farmaci (Drug Discovery)"},
    {"topic": "Tecniche di mitigazione delle allucinazioni nei Large Language Models per applicazioni mediche"},

    # Fisica e Nuove Tecnologie
    {"topic": "Recenti avanzamenti tecnologici nella fusione nucleare a confinamento magnetico"},
    {"topic": "L'impatto dei futuri computer quantistici sulla sicurezza della crittografia asimmetrica"},

    # Ingegneria dei Materiali / Hardware
    {"topic": "Applicazioni industriali dei nanomateriali in grafene per le batterie a stato solido"},
    {"topic": "Sviluppo e sfide delle interfacce cervello-computer (BCI) non invasive nel 2026"},

    # Spazio ed Esplorazione
    {"topic": "Le tecnologie alla base dei nuovi rover autonomi per l'esplorazione della superficie marziana"}
]


# ==========================================
# 2. PROMPT E VALUTATORE LANGSMITH "ANTIPROIETTILE"
# ==========================================
def langsmith_evaluator(run, example) -> dict:
    final_state = run.outputs

    if not final_state:
        return {"key": "grounding_score", "score": 0.0, "comment": "Errore: Lo stato finale dell'agente è vuoto."}

    # 1. Recupero SICURO del Topic (lo prendiamo direttamente dall'input del test LangSmith)
    topic = example.inputs["topic"]

    # 2. Recupero SICURO di Articolo e Fonti dalla cronologia dei messaggi
    messages = final_state.get("messages", [])

    tool_outputs = []
    ai_messages = []

    for m in messages:
        # Gestione sicura dell'attributo type (funziona sia con oggetti BaseMessage che con dizionari)
        m_type = getattr(m, 'type', m.get('type', '') if isinstance(m, dict) else '')
        content = getattr(m, 'content', m.get('content', '') if isinstance(m, dict) else str(m))

        # Se è l'output di un tool (Internet Search, RAG, Matcher), lo salviamo nelle Fonti
        if m_type == "tool" and content.strip():
            tool_outputs.append(content)

        # Se è un messaggio AI bello lungo (ignoriamo i piccoli "Ok, fatto"), lo consideriamo come bozza/articolo
        elif m_type == "ai" and content.strip() and len(content) > 50:
            ai_messages.append(content)

    # Uniamo tutti i risultati dei tool per formare il Contesto/Sources
    sources = "\n\n---\n\n".join(tool_outputs) if tool_outputs else "Nessuna fonte trovata dai tool."

    # L'articolo finale sarà l'ultimo messaggio corposo dell'AI
    final_article = ai_messages[-1] if ai_messages else "Nessun articolo generato."

    # Prompt aggiornato per il giudice
    evaluator_prompt = f"""
    Sei un valutatore esperto incaricato di analizzare un post per un blog.
    Valuta:
    1. **Analisi Qualitativa**: Il post è coerente con il Topic "{topic}"?
    2. **Grounding**: Le affermazioni nell'articolo sono supportate dalle Fonti recuperate dai tool?

    Rispondi SOLO con un numero float tra 0.0 (pessimo) e 1.0 (perfetto) seguito da un a capo e un breve report critico.
    """

    # Chiamiamo il tuo LLM As a Judge
    try:
        score, reason = call_llm_judge(evaluator_prompt, final_article, sources)
    except Exception as e:
        return {"key": "grounding_score", "score": 0.0, "comment": f"Errore interno del giudice: {e}"}

    try:
        parsed_score = float(score)
    except ValueError:
        parsed_score = 0.0
        reason = f"Errore di parsing del punteggio: {score}\n\n" + reason

    return {"key": "grounding_score", "score": parsed_score, "comment": reason}


# ==========================================
# 3. FUNZIONE DI AUTOMAZIONE E BYPASS HITL
# ==========================================
import time  # <-- Assicurati di avere questo import in cima al file


# ==========================================
# 3. FUNZIONE DI AUTOMAZIONE E BYPASS HITL
# ==========================================
def automated_agent_run(inputs: dict) -> dict:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # PROMPT AGGIORNATO E BLINDATO CONTRO I LOOP
    test_prompt = (
        f"⚠️ REGOLA DI TEST AUTOMATICO ⚠️\n"
        f"Scrivi un articolo sul topic: '{inputs['topic']}'.\n"
        f"Fai le tue ricerche ma poi, per redigere il testo finale, DEVI OBBLIGATORIAMENTE "
        f"usare il tool 'write_an_article' passandogli il contenuto.\n"
        f"VIETATO rispondere testualmente in chat dicendo 'Ho finito' o 'Ecco l'articolo'. "
        f"L'UNICO modo per concludere è chiamare la funzione 'write_an_article'."
    )

    app.invoke(
        Command(
            update={
                "messages": [HumanMessage(content=test_prompt)],
                "classification_decision": None
            }
        ),
        config=config
    )

    while True:
        current_state = app.get_state(config)

        if not current_state.next:
            break

        paused_node = current_state.next[0]
        print(f"🔄 Bypass automatico HITL in '{paused_node}' (Thread: {thread_id[:8]})...")

        print("⏳ Pausa di 15 secondi per rispettare i limiti di OpenAI...")
        time.sleep(15)

        # Un resume più netto per non far ripartire l'LLM a chiacchierare
        resume_message = "L'articolo 1 va bene."
        app.invoke(Command(resume=resume_message), config)

    final_state = app.get_state(config).values
    return final_state


# ==========================================
# 4. ESECUZIONE DELLA SUITE
# ==========================================
if __name__ == "__main__":
    print("🌐 Inizializzazione del client LangSmith...")
    client = Client()
    dataset_name = "CCAI-Blog-Topics-v3"  # Modificato in v3 per un test pulito

    try:
        client.read_dataset(dataset_name=dataset_name)
        print(f"✅ Dataset '{dataset_name}' trovato su LangSmith.")
    except Exception:
        print(f"⚠️ Dataset non trovato. Creazione di '{dataset_name}' in corso...")
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Test automatizzato per il progetto CCAI 2026"
        )
        client.create_examples(
            inputs=dataset_topics,
            dataset_id=dataset.id
        )
        print("✅ Dataset creato con successo.")

    print("🚀 Avvio della suite di test automatizzata (lascialo finire senza interromperlo)...")

    experiment_results = evaluate(
        automated_agent_run,
        data=dataset_name,
        evaluators=[langsmith_evaluator],
        experiment_prefix="CCAI-Auto-Test",
        description="Esecuzione batch robusta basata sui messaggi"
    )

    print("✅ Test completati! Vai alla dashboard di LangSmith per vedere i risultati.")