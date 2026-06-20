from langsmith.evaluation import evaluate

# Definisci il prompt per il giudice LLM
evaluator_prompt = """
Sei un valutatore esperto. Il tuo compito è analizzare un articolo di un blog.
Ti fornirò il CONTESTO (fonti web e nodi del Knowledge Graph) e l'ARTICOLO FINALE.

Devi valutare due aspetti:
1. GROUNDING: Tutte le affermazioni fattuali nell'articolo sono presenti nel CONTESTO?
2. CITATIONS: L'articolo cita esplicitamente le fonti?

Rispondi con un punteggio tra 0.0 e 1.0 e una breve spiegazione.
"""


def grounding_evaluator(run, example) -> dict:
    # Estrai l'articolo finale e il contesto recuperato dai log della run
    final_article = run.outputs["final_article"]
    retrieved_context = run.outputs["context_used"]

    # Chiama l'LLM giudice (es. gpt-4o-mini o un modello locale) con il prompt di valutazione
    score, reason = call_llm_judge(evaluator_prompt, final_article, retrieved_context)

    return {"key": "grounding_score", "score": score, "comment": reason}

# Esecuzione della suite di test sui post generati
# evaluate(
#     target=your_langgraph_pipeline,
#     data="Blog_Topics_Dataset",
#     evaluators=[grounding_evaluator]
# )