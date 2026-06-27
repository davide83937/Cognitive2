import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Sostituisci con il path reale della cartella dove hai salvato i pesi del modello fine-tuned
MODEL_PATH = "./pesi_lora_v1_stabili/checkpoint-525"

print("⏳ Caricamento del modello Text-to-Cypher in corso...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"  # Usa la GPU se disponibile sul tuo PC locale
)
print("✅ Modello caricato!")


def generate_cypher(query_testuale: str) -> str:
    """
    Genera una query Cypher deterministica usando il modello fine-tuned.
    Usa la logica di slicing per garantire un output pulito.
    """
    prompt_text = """Genera SOLO la query Neo4j Cypher. Non aggiungere spiegazioni o formattazione markdown.
REGOLE RIGIDE:
1. Restituisci in RETURN SOLO le variabili esplicitamente richieste.
2. Le date ('date') appartengono ESCLUSIVAMENTE al nodo Post (p.date). I Claim non hanno date.
3. Mantieni i percorsi ottimali e diretti.

SCHEMA CONSENTITO:
- Nodi: Post {title, date}, Topic {name}, Claim {text}, Source {name}, Documentation {text}
- Relazioni Post: (Post)-[:COVERS]->(Topic), (Post)-[:USES]->(Documentation), (Post)-[:EXTRACTS]->(Claim)
- Relazioni Topic: (Topic)-[r:RELATED_TO]->(Topic) dove r.type DEVE ESSERE 'PREREQUISITO', 'CONTRASTO', 'ESTENSIONE', 'SIMILARE', 'SOTTO_CATEGORIA' o 'APPLICAZIONE'.

ESEMPI DI RIFERIMENTO:
Utente: Seleziona i post di aprile 2026 che coprono il topic 'Docker'.
Cypher: MATCH (p:Post)-[:COVERS]->(t:Topic) WHERE p.date STARTS WITH '2026-04' AND t.name =~ '(?i).*docker.*' RETURN p.title
Utente: Topic simili a 'RAG' senza contrasti.
Cypher: MATCH (t1:Topic)-[:RELATED_TO {type: 'SIMILARE'}]->(t2:Topic) WHERE t2.name =~ '(?i).*rag.*' AND NOT (t1)-[:RELATED_TO {type: 'CONTRASTO'}]-(:Topic) RETURN t1.name

Richiesta: """

    # 1. ASSEMBLAGGIO DEL PROMPT
    full_prompt = prompt_text + query_testuale + "\nCypher: "

    # 2. TOKENIZZAZIONE
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True).to(model.device)

    # 3. GENERAZIONE GUIDATA (Output 100% deterministico)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # Nessuna casualità
            temperature=None,  # Annulla default
            top_p=None,  # Annulla default
            top_k=None,  # Annulla default
            repetition_penalty=1.2,  # Previene loop
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 4. ESTRAZIONE DEL SOLO OUTPUT GENERATO (Tensor slicing)
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    risultato_pulito = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # 5. SICUREZZA AGGIUNTIVA: Rimuove eventuali residui di Markdown se il modello li genera comunque
    risultato_pulito = risultato_pulito.replace("```cypher", "").replace("```", "").strip()

    return risultato_pulito