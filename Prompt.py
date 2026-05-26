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
ad esempio se il topic è "elettricità" oppure "droni", bisogna essere più specifici
- reject, Se l'argomento non ha nulla a che vedere con la scienza o la tecnologia"""