import os
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.messages import HumanMessage

from Models import get_llm
from Prompt import triage_system_prompt
from Schemas import RouterSchema

# Importa le tue funzioni reali. (Adatta gli import al tuo progetto)
# from Nodes import get_llm, triage_system_prompt
# from Schemas import RouterSchema

client = Client()

# ==========================================
# 1. CREAZIONE DEL DATASET (GROUND TRUTH)
# ==========================================
dataset_name = "Triage_Router_Dataset"

# Creiamo il dataset su LangSmith solo se non esiste già
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name, description="Valutazione del prompt di routing")

    # Esempi di test: inseriamo Argomento (input) e Classificazione Attesa (output)
    examples = [
        # Casi da ACCEPT (specifici e in target)
        {"inputs": {"topic": "Utilizzo di LangGraph per estrarre entità da un PDF"}, "outputs": {"expected": "accept"}},
        {"inputs": {"topic": "Come configurare i timer PWM su STM32 per un braccio robotico"},
         "outputs": {"expected": "accept"}},

        # Casi da REFINE (troppo vaghi)
        {"inputs": {"topic": "Robotica"}, "outputs": {"expected": "refine"}},
        {"inputs": {"topic": "Programmazione in Python"}, "outputs": {"expected": "refine"}},

        # Casi da REJECT (fuori tema)
        {"inputs": {"topic": "Ricetta della pasta alla carbonara"}, "outputs": {"expected": "reject"}},
        {"inputs": {"topic": "Come allenare il petto a corpo libero nel calisthenics"},
         "outputs": {"expected": "reject"}},
    ]

    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=dataset.id,
    )


# ==========================================
# 2. DEFINIZIONE DEL TARGET (IL "CERVELLO")
# ==========================================
# Questa funzione isola solo la chiamata LLM del tuo router reale
def run_triage_llm(inputs: dict) -> dict:
    topic = inputs["topic"]

    # Inizializza il tuo LLM reale e passagli il prompt reale
    llm = get_llm().with_structured_output(RouterSchema)

    result = llm.invoke([
        {"role": "system", "content": triage_system_prompt},
        {"role": "user", "content": topic},
    ])

    # Restituiamo ciò che ha deciso il modello
    return {
        "predicted_classification": result.classification,
        "ragionamento": result.ragionamento
    }


# ==========================================
# 3. DEFINIZIONE DELLA METRICA DI VALUTAZIONE
# ==========================================
# Una semplice funzione che confronta l'output atteso con quello predetto
def exact_match_evaluator(run, example) -> dict:
    expected = example.outputs["expected"]
    predicted = run.outputs["predicted_classification"]

    # Diamo punteggio 1 se ha indovinato, 0 se ha sbagliato
    score = 1 if expected == predicted else 0
    return {"key": "accuracy", "score": score}


# ==========================================
# 4. ESECUZIONE DELL'ESPERIMENTO
# ==========================================
if __name__ == "__main__":
    print("Avvio della valutazione su LangSmith...")

    experiment_results = evaluate(
        run_triage_llm,
        data=dataset_name,
        evaluators=[exact_match_evaluator],
        experiment_prefix="Triage-Eval-V1",
        # max_concurrency=2 # Decommenta se ricevi errori di Rate Limit dalle API
    )

    print("Valutazione completata! Controlla la dashboard di LangSmith.")