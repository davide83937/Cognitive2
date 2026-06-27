@tool
def find_first_available_date_tool(data_partenza: str = None) -> str:
    """
    Trova la prima data disponibile nel Knowledge Graph per pubblicare un articolo.
    La programmazione avviene rigorosamente solo di Lunedì, Mercoledì, Venerdì e Domenica.
    Se 'data_partenza' non è fornita, riparte dall'ultimo articolo pubblicato nel database.
    Una giornata è piena se ci sono già 3 articoli schedulati.
    """
    if not graph:
        return "Errore: Database Neo4j non connesso."

    # Se l'LLM non ha una data specifica di partenza, peschiamo l'ultima assoluta dal DB
    if not data_partenza or data_partenza == 'null':
        base_date_str = get_latest_scheduled_date_from_db()
    else:
        base_date_str = data_partenza

    # Troviamo il prossimo giorno consentito dal palinsesto
    giorno_corrente = get_next_fixed_publish_date(base_date_str)

    query = get_post_count_by_date_query()

    # Cerchiamo in avanti saltando ai soli giorni consentiti
    for _ in range(50):  # Limite di tentativi per non creare loop infiniti
        data_str = giorno_corrente.strftime("%Y-%m-%d")

        try:
            risultato = graph.query(query, params={"date": data_str})
            articoli_presenti = risultato[0]["current_count"] if risultato else 0

            if articoli_presenti < 3:
                return f"La prima data disponibile in palinsesto (Lun, Mer, Ven, Dom) nel Knowledge Graph è: {data_str}"
            # Se il giorno è pieno, saltiamo al PROSSIMO giorno di palinsesto
            giorno_corrente = get_next_fixed_publish_date(data_str)

        except Exception as e:
            return f"Errore durante l'interrogazione del KG: {e}"

    return "Non ci sono date disponibili nel palinsesto."


@tool
def check_specific_date_tool(data_target: str) -> str:
    """
    Verifica se una data specifica è disponibile nel Knowledge Graph per schedulare un articolo.
    L'argomento 'data_target' deve essere rigorosamente 'YYYY-MM-DD'.
    Verifica anche che la data rispetti il palinsesto: Lunedì, Mercoledì, Venerdì, Domenica.
    Una giornata è piena se ci sono già 3 articoli schedulati.
    """
    if not graph:
        return "Errore: Database Neo4j non connesso."
    # 1. Verifica che il giorno della settimana sia consentito
    giorni_pubblicazione = [0, 2, 4, 6]
    giorni_nomi = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]
    try:
        target_date_obj = datetime.strptime(data_target, "%Y-%m-%d").date()
    except ValueError:
        return "Errore: Formato data non valido, usa YYYY-MM-DD."
    giorno_settimana = target_date_obj.weekday()
    if giorno_settimana not in giorni_pubblicazione:
        return (f"Rifiutato: La data {data_target} è un {giorni_nomi[giorno_settimana]}. "
                f"Il blog pubblica solo di Lunedì, Mercoledì, Venerdì e Domenica. "
                f"Prova a cercare la prima data disponibile.")
    # 2. Se è un giorno consentito, verifica l'occupazione in Neo4j
    query = "MATCH (p:Post {date: $date}) RETURN count(p) AS current_count"
    try:
        risultato = graph.query(query, params={"date": data_target})
        current_count = risultato[0]["current_count"] if risultato else 0
        if current_count < 3:
            return (f"La data {data_target} ({giorni_nomi[giorno_settimana]}) è DISPONIBILE in palinsesto. "
                    f"Attualmente ci sono {current_count} articoli pianificati.")
        else:
            return (f"La data {data_target} è OCCUPATA. "
                    f"Ci sono già {current_count} articoli pianificati (limite massimo).")
    except Exception as e:
        return f"Errore durante l'interrogazione del KG: {e}"