from langchain_core.tools import tool
from Notion_Stuff import trova_prima_data_disponibile, controlla_disponibilita_data


@tool
def write_an_article(about: str, author: str, content: str):
    """Write an article about the topic given by user, author is IA"""
    return f"My article about: {about}. \n{content} \n Written by {author}"


@tool
def find_first_available_date_tool(data_partenza: str = None) -> str:
    """
    Trova la prima data disponibile su Notion per pubblicare un articolo.
    Accetta opzionalmente una 'data_partenza' (YYYY-MM-DD). Se non fornita, cerca da oggi.
    Restituisce la data trovata o un messaggio se il calendario è pieno.
    """
    data_trovata = trova_prima_data_disponibile(data_partenza)
    if data_trovata:
        return f"La prima data disponibile trovata su Notion è: {data_trovata}"
    else:
        return "Non ci sono date disponibili su Notion per l'arco di tempo richiesto."


@tool
def check_specific_date_tool(data_target: str) -> str:
    """
    Verifica se una data specifica è disponibile su Notion per schedulare un articolo.
    L'argomento 'data_target' deve essere rigorosamente nel formato stringa 'YYYY-MM-DD' (es. '2026-06-01').
    Da usare quando l'utente propone una data precisa o chiede se un determinato giorno è libero.
    """
    risultato = controlla_disponibilita_data(data_target)

    if risultato is None:
        return f"Impossibile verificare la disponibilità per la data {data_target} a causa di un errore di comunicazione con Notion."

    if risultato["is_available"]:
        return (f"La data {data_target} è DISPONIBILE su Notion. "
                f"Attualmente ci sono {risultato['current_count']} articoli pianificati per questo giorno.")
    else:
        return (f"La data {data_target} è OCCUPATA/PIENA. "
                f"Ci sono già {risultato['current_count']} articoli pianificati, raggiungendo il limite massimo.")