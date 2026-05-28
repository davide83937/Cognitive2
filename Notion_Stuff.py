from collections import Counter
from datetime import timedelta, datetime

import requests

from Models import get_notion_token, get_notion_db_id


def add_row_to_notion(title, date, author, text):
    print(f"Tentativo di aggiunta di: '{title}'...")

    # Per aggiungere righe, l'endpoint corretto è quello delle "pages"
    url = "https://api.notion.com/v1/pages"

    headers = {
        "Authorization": f"Bearer {get_notion_token()}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"  # Fondamentale per l'invio del JSON
    }

    # La struttura del corpo della richiesta (Payload)
    payload = {
        "parent": {
            "database_id": get_notion_db_id()
        },
        "properties": {
            # 1. Colonna Titolo (Obbligatoria)
            "Title": {
                "title": [{"text": {"content": title}}]
            },

            # 2. Colonna Data (Tipo: Date su Notion)
            "Date": {
                "date": {
                    "start": date  # Formato YYYY-MM-DD
                }
            },

            # 4. Colonna Autore (Tipo: Select su Notion)
            "Author": {
                "select": {
                    "name": author  # L'opzione della lista
                }
            },

            # 5. Colonna Testo/Nota (Tipo: Text / Rich Text su Notion)
            "Text": {
                "rich_text": [
                    {
                        "text": {
                            "content": text
                        }
                    }
                ]
            }
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            print(f"✅ RIGA AGGIUNTA CON SUCCESSO!")
            # Puoi estrarre l'URL della riga appena creata se ti serve
            res_data = response.json()
            print(f"Elemento creato qui: {res_data.get('url')}")
        else:
            print(f"❌ ERRORE {response.status_code}:")
            print(response.text)

    except Exception as e:
        print(f"❌ ERRORE DI ESECUZIONE: {e}")


def controlla_disponibilita_data(data_target, max_articoli=3):
    """
    Controlla se una specifica data ha meno di 'max_articoli' schedulati.
    data_target deve essere nel formato 'YYYY-MM-DD' (es. '2026-06-01').
    """
    print(f"Agente: Controllo disponibilità per la data {data_target}...")

    # L'endpoint per interrogare (query) un database
    url = f"https://api.notion.com/v1/databases/{get_notion_db_id()}/query"

    headers = {
        "Authorization": f"Bearer {get_notion_token()}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

    # Costruiamo il filtro: "Trovami tutte le righe dove la colonna 'Data' è uguale a data_target"
    # ATTENZIONE: "Data" deve essere il nome esatto della tua colonna su Notion
    payload = {
        "filter": {
            "property": "Date",
            "date": {
                "equals": data_target
            }
        }
    }

    try:
        # Nota: usiamo POST per fare una query
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            dati = response.json()

            # La chiave 'results' contiene una lista di tutte le pagine trovate
            articoli_trovati = len(dati.get("results", []))

            disponibile = articoli_trovati < max_articoli

            if disponibile:
                print(f"✅ DATA DISPONIBILE: Ci sono {articoli_trovati}/{max_articoli} articoli schedulati.")
            else:
                print(
                    f"❌ DATA PIENA: Il limite di {max_articoli} articoli è già stato raggiunto ({articoli_trovati} presenti).")

            # Restituiamo i dati in modo che l'agente possa usarli per prendere decisioni nel workflow
            return {
                "is_available": disponibile,
                "current_count": articoli_trovati,
                "target_date": data_target
            }

        else:
            print(f"❌ ERRORE API {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"❌ ERRORE DI ESECUZIONE: {e}")
        return None


def trova_prima_data_disponibile(data_partenza=None, max_articoli=3):
    """
    Trova la prima data (da data_partenza in poi) che ha meno di 'max_articoli' schedulati.
    Se data_partenza non è fornita, usa la data di oggi.
    """
    # Se non specifichiamo una data, partiamo da oggi
    if data_partenza is None:
        oggi = datetime.now().date()
    else:
        # Assumiamo che data_partenza sia una stringa 'YYYY-MM-DD'
        oggi = datetime.strptime(data_partenza, "%Y-%m-%d").date()

    print(f"Agente: Ricerca prima data disponibile a partire dal {oggi}...")

    url = f"https://api.notion.com/v1/databases/{get_notion_db_id()}/query"

    headers = {
        "Authorization": f"Bearer {get_notion_token()}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

    # 1. Chiediamo a Notion TUTTI gli articoli schedulati da 'oggi' in poi
    payload = {
        "filter": {
            "property": "Date",  # Assicurati che questo sia il nome esatto della colonna
            "date": {
                "on_or_after": oggi.strftime("%Y-%m-%d")
            }
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            dati = response.json()
            pagine = dati.get("results", [])

            # 2. Estraiamo tutte le date occupate e le contiamo
            # Otterremo qualcosa tipo: {"2026-05-28": 3, "2026-05-29": 1, "2026-05-31": 3}
            date_occupate = []
            for pagina in pagine:
                # Navighiamo la struttura JSON di Notion per estrarre la data
                # Aggiungiamo un check nel caso una riga abbia la colonna Data vuota
                campo_data = pagina.get("properties", {}).get("Data", {}).get("date")
                if campo_data and campo_data.get("start"):
                    date_occupate.append(campo_data["start"])

            conteggio_giornaliero = Counter(date_occupate)

            # 3. Scorriamo i giorni a partire da oggi finché non troviamo un posto libero
            giorno_corrente = oggi
            limite_ricerca_giorni = 365  # Evitiamo loop infiniti

            for _ in range(limite_ricerca_giorni):
                data_str = giorno_corrente.strftime("%Y-%m-%d")

                # Quanti articoli ci sono in questo giorno? (0 se la data non esiste nel Counter)
                articoli_presenti = conteggio_giornaliero.get(data_str, 0)

                if articoli_presenti < max_articoli:
                    print(f"✅ TROVATA DATA: {data_str} (Articoli presenti: {articoli_presenti}/{max_articoli})")
                    return data_str

                # Se è pieno, passiamo al giorno successivo
                giorno_corrente += timedelta(days=1)

            print("❌ ERRORE: Nessuna data disponibile trovata nell'arco di un anno.")
            return None

        else:
            print(f"❌ ERRORE API {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"❌ ERRORE DI ESECUZIONE: {e}")
        return None