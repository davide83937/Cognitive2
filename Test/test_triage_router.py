from langsmith import Client
from langsmith.evaluation import evaluate


from Models import get_llm
from Prompt import triage_system_prompt
from Schemas import RouterSchema



client = Client()
dataset_name = "Triage_Router_Dataset"


if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name, description="Valutazione del prompt di routing")


    examples = [
        {"inputs": {"topic": "Utilizzo di LangGraph per estrarre entità da un PDF"}, "outputs": {"expected": "accept"}},
        {"inputs": {"topic": "Come configurare i timer PWM su STM32 per un braccio robotico"},
         "outputs": {"expected": "accept"}},

        {"inputs": {"topic": "Robotica"}, "outputs": {"expected": "refine"}},
        {"inputs": {"topic": "Programmazione in Python"}, "outputs": {"expected": "refine"}},

        {"inputs": {"topic": "Ricetta della pasta alla carbonara"}, "outputs": {"expected": "reject"}},
        {"inputs": {"topic": "Come allenare il petto a corpo libero nel calisthenics"},
         "outputs": {"expected": "reject"}},
    ]

    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=dataset.id,
    )


def run_triage_llm(inputs: dict) -> dict:
    topic = inputs["topic"]
    llm = get_llm().with_structured_output(RouterSchema)

    result = llm.invoke([
        {"role": "system", "content": triage_system_prompt},
        {"role": "user", "content": topic},
    ])

    return {
        "predicted_classification": result.classification,
        "ragionamento": result.ragionamento
    }


def exact_match_evaluator(run, example) -> dict:
    expected = example.outputs["expected"]
    predicted = run.outputs["predicted_classification"]

    score = 1 if expected == predicted else 0
    return {"key": "accuracy", "score": score}



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