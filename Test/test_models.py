import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class EvaluationResult(BaseModel):
    score: float = Field(
        description="Punteggio: 1.0 se l'articolo rispetta i criteri (grounding/citazioni), 0.0 se fallisce.")
    reasoning: str = Field(description="Breve spiegazione del motivo per cui è stato assegnato il punteggio.")


def call_llm_judge(evaluator_prompt: str, final_article: str, retrieved_context: str) -> tuple[float, str]:
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    structured_judge = llm.with_structured_output(EvaluationResult)

    user_content = (
        f"--- CONTESTO RECUPERATO DAI TOOL ---\n{retrieved_context}\n\n"
        f"--- ARTICOLO GENERATO ---\n{final_article}"
    )

    result = structured_judge.invoke([
        {"role": "system", "content": evaluator_prompt},
        {"role": "user", "content": user_content}
    ])

    return result.score, result.reasoning