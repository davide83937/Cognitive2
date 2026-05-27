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

def get_accept_prompt(last_input: str):
    accept_prompt = f"""Sei un giornalista che scrive articoli riguardanti un argomento che ti viene passato in input.
    In questo caso l'argomento su cui dovrai scrivere un articolo è: '{last_input}'."""
    return accept_prompt

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