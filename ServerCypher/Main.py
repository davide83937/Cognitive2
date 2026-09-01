import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

app = Flask(__name__)

# Inserisci i percorsi validi sul secondo PC
# Puoi copiare la cartella 'checkpoint-525' direttamente su questo PC tramite chiavetta o rete
LORA_WEIGHTS_PATH = r"C:\Users\david\OneDrive\Desktop\Università\COGNITIVE COMPUTING AND ARTIFICIAL INTELLIGENCE\FINE_T\checkpoint-525-test5"
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print("⏳ Caricamento del Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_WEIGHTS_PATH)

print("⏳ Configurazione NF4 per QLoRA...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("⏳ Caricamento del modello base Qwen-7B sulla RTX 3070...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

print("⏳ Iniezione dei pesi QLoRA locali...")
model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
print("✅ Modello Text-to-Cypher pronto sulla RTX 3070!")


@app.route('/generate_cypher', methods=['POST'])
def generate_cypher_api():
    data = request.json
    prompt = data.get("prompt", "")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.01,  # uso lo stesso valore conservativo di Kaggle
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    risultato_pulito = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    risultato_pulito = risultato_pulito.replace("```cypher", "").replace("```", "").strip()

    return jsonify({"cypher_query": risultato_pulito})


if __name__ == '__main__':
    # '0.0.0.0' permette al server di accettare connessioni dagli altri PC della rete locale
    app.run(host='0.0.0.0', port=5000)