import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langsmith.evaluation import evaluate
from langsmith import Client

# Importiamo i TUOI tool e il dataset
from post_evaluator import dataset_topics
from base import get_tools_by_name


# --- 1. SCHEMI (Definiti solo per questo test) ---
class StructuredArticle(BaseModel):
    title: str = Field(description="Titolo accattivante dell'articolo")
    subtitle: str = Field(description="Sottotitolo esplicativo")
    introduction: str = Field(description="Introduzione al problema o al tema")
    technical_body: str = Field(description="Corpo principale, formattato in Markdown")
    conclusion: str = Field(description="Sintesi finale")
    tags: list[str] = Field(description="Lista di 3-5 tag")


class QualitativeReport(BaseModel):
    qualita_post: str = Field(description="Analisi del tono e della struttura.")
    analisi_fonti: str = Field(description="Analisi del grounding, citando KG, RAG e Web.")
    giudizio_finale: str = Field(description="Caso di successo o fallimento?")
    score: float = Field(description="Voto da 0.0 a 1.0.")


# --- 2. IL TARGET SPERIMENTALE (Ricostruisce la tua logica K-RAG) ---
def run_experimental_generation(inputs: Dict[str, Any]) -> Dict[str, Any]:
    topic = inputs.get("topic")
    print(f"\n🔍 Avvio ricerca completa per: {topic}")

    # A. KNOWLEDGE GRAPH (Matcher + Enhanced Context)
    kg_context = ""
    try:
        matcher = get_tools_by_name["intelligent_topic_matcher"]
        topic_matchati = matcher.invoke({"topic": topic})

        if topic_matchati and "Nessun" not in str(topic_matchati):
            # Se trova più topic, li cicliamo (come richiede la tua Regola 2)
            lista_topic = [t.strip() for t in str(topic_matchati).split(",")]
            enhancer = get_tools_by_name["get_enhanced_topic_context"]
            for t in lista_topic:
                dati_nodo = enhancer.invoke({"topic": t})
                kg_context += f"\n[KG - {t}]: {dati_nodo}"
        else:
            # Fallback
            enhancer = get_tools_by_name["get_enhanced_topic_context"]
            kg_context = enhancer.invoke({"topic": topic})
    except Exception as e:
        kg_context = f"Errore KG: {e}"

    # B. RAG LOCALE (Document Retriever)
    rag_context = ""
    try:
        # Arricchiamo la query del RAG con le info del KG (Regola 4)
        rag_query = f"{topic}. Contesto noto: {kg_context[:100]}"
        rag_tool = get_tools_by_name["rag_document_retriever"]
        # Adatta gli argomenti se il tuo RAG accetta parametri diversi
        rag_context = rag_tool.invoke(rag_query)
    except Exception as e:
        rag_context = f"Errore RAG: {e}"

    # C. RICERCA WEB (Tavily)
    web_context = ""
    try:
        web_tool = get_tools_by_name["verified_internet_search"]
        web_context = web_tool.invoke({"query": topic})
    except Exception as e:
        web_context = f"Errore Web: {e}"

    # ASSEMBLAGGIO DEL CONTESTO TOTALE
    contesto_totale = (
        f"--- FONTE KNOWLEDGE GRAPH (Neo4j) ---\n{kg_context}\n\n"
        f"--- FONTE DOCUMENTI LOCALI (RAG) ---\n{rag_context}\n\n"
        f"--- FONTE WEB (Tavily) ---\n{web_context}"
    )

    # D. GENERAZIONE STRUTTURATA (Il nuovo approccio)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2).with_structured_output(StructuredArticle)

    prompt = f"""
    Sei un Copywriter Tecnico AI. Scrivi un articolo di altissima qualità su: '{topic}'.

    BASATI ESCLUSIVAMENTE SU QUESTO CONTESTO (che include Storico, Manuali RAG e Notizie Web):
    {contesto_totale}

    Nel testo cita le fonti in modo naturale (es. "Secondo i dati in nostro possesso (KG)...", oppure "Studi recenti (RAG) indicano...").
    """

    print(f"✍️ Stesura articolo in corso...")
    articolo = llm.invoke(prompt)

    testo_post = (
        f"# {articolo.title}\n"
        f"*{articolo.subtitle}*\n\n"
        f"### Introduzione\n{articolo.introduction}\n\n"
        f"### Approfondimento\n{articolo.technical_body}\n\n"
        f"### Conclusioni\n{articolo.conclusion}\n\n"
        f"*Tags:* {', '.join(articolo.tags)}"
    )

    return {"final_post": testo_post, "retrieved_context": contesto_totale}


# --- 3. EVALUATOR ---
def qualitative_analysis_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    post = run.outputs.get("final_post")
    context = run.outputs.get("retrieved_context")

    llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(QualitativeReport)
    prompt = f"Analizza questo post valutando come ha unito le fonti (KG, RAG, WEB):\n\nCONTESTO:{context}\n\nPOST:{post}"
    report = llm.invoke(prompt)

    return {
        "key": "Qualitative_Analysis",
        "score": report.score,
        "comment": f"Analisi Post: {report.qualita_post}\n\nFonti: {report.analisi_fonti}\n\nGiudizio: {report.giudizio_finale}"
    }


# --- 4. ESECUZIONE ---
if __name__ == "_main_":
    client = Client()
    dataset_name = "Experimental_Generation_Test"

    try:
        ds = client.create_dataset(dataset_name=dataset_name)
    except:
        ds = client.read_dataset(dataset_name=dataset_name)

    examples = list(client.list_examples(dataset_id=ds.id))
    if not examples:
        client.create_examples(inputs=[{"topic": d["topic"]} for d in dataset_topics], dataset_id=ds.id)

    print("🚀 Avvio Test Sperimentale con Logica K-RAG su LangSmith...")
    evaluate(
        run_experimental_generation,
        data=dataset_name,
        evaluators=[qualitative_analysis_evaluator],
        experiment_prefix="Strutturato-Completo-KRAG"
    )