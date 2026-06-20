import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# 1. Definiamo lo schema esatto che vogliamo che il giudice restituisca
class EvaluationResult(BaseModel):
    score: float = Field(
        description="Punteggio: 1.0 se l'articolo rispetta i criteri (grounding/citazioni), 0.0 se fallisce.")
    reasoning: str = Field(description="Breve spiegazione del motivo per cui è stato assegnato il punteggio.")


def call_llm_judge(evaluator_prompt: str, final_article: str, retrieved_context: str) -> tuple[float, str]:
    # 2. Inizializziamo il modello che hai scelto
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 3. Forziamo il modello a rispondere usando lo schema definito
    structured_judge = llm.with_structured_output(EvaluationResult)

    # 4. Assembliamo i dati per il prompt
    user_content = (
        f"--- CONTESTO RECUPERATO DAI TOOL ---\n{retrieved_context}\n\n"
        f"--- ARTICOLO GENERATO ---\n{final_article}"
    )

    # 5. Invochiamo il modello
    result = structured_judge.invoke([
        {"role": "system", "content": evaluator_prompt},
        {"role": "user", "content": user_content}
    ])

    return result.score, result.reasoning