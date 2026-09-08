#!/usr/bin/env python3
"""
Holt zu einer Liste von Wikidata-QIDs die Labels (Namen) und optional
Wikipedia-Titel per SPARQL vom Wikidata Query Service (WDQS).

Eingabe:  eine Textdatei mit einer QID pro Zeile (z.B. "Q42", ohne "wd:")
Ausgabe:  eine CSV-Datei mit qid,label_de,label_en,dewiki_title,enwiki_title

Funktionsweise:
- QIDs werden in Batches gruppiert (Default: 800 pro Query, konservativ
  gewaehlt wegen der 60s-Prozesszeit-Regel von WDQS).
- Fortschritt wird laufend in die Ausgabedatei geschrieben (append),
  bereits verarbeitete QIDs werden bei einem Neustart uebersprungen.
- Bei 429/503 wird mit exponentiellem Backoff gewartet und Retry-After
  respektiert, falls vorhanden.
- Ein aussagekraeftiger User-Agent ist Pflicht (Wikimedia UA Policy).

Nutzung:
    python wikidata_labels_fetch.py qids.txt output.csv \
        --user-agent "MeinProjekt/1.0 (kontakt@example.org)"
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import requests

WDQS_ENDPOINT = "https://query.wikidata.org/sparql"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wikidata_labels_fetch")


def read_qids(path: str) -> list[str]:
    """Liest QIDs aus einer Textdatei, eine pro Zeile. Normalisiert auf 'Q123'."""
    qids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid = line.strip()
            if not qid:
                continue
            if not qid.upper().startswith("Q"):
                qid = "Q" + qid
            qids.append(qid.upper())
    return qids


def load_already_done(output_path: str) -> set[str]:
    """Liest bereits vorhandene QIDs aus der Ausgabedatei (fuer Resume nach Abbruch)."""
    done = set()
    p = Path(output_path)
    if p.exists():
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row["qid"])
    return done


def build_query(qids: list[str], langs: str) -> str:
    """Baut die SPARQL-Query fuer einen Batch von QIDs."""
    values = " ".join(f"wd:{q}" for q in qids)
    primary_lang = langs.split(",")[0]
    return f"""
    SELECT ?item ?itemLabel
           ?dewikiTitle ?enwikiTitle
    WHERE {{
      VALUES ?item {{ {values} }}
      OPTIONAL {{
        ?dewikiSitelink schema:about ?item ;
                         schema:isPartOf <https://de.wikipedia.org/> ;
                         schema:name ?dewikiTitle .
      }}
      OPTIONAL {{
        ?enwikiSitelink schema:about ?item ;
                         schema:isPartOf <https://en.wikipedia.org/> ;
                         schema:name ?enwikiTitle .
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{langs}". }}
    }}
    """


RETRYABLE_STATUS = (429, 500, 502, 503, 504)


def run_query(session: requests.Session, query: str, user_agent: str,
              max_retries: int = 8) -> dict:
    """Fuehrt eine SPARQL-Query aus, mit Retry/Backoff bei transienten Fehlern.

    Wiederholt wird bei Netzwerkfehlern und HTTP 429/500/502/503/504 -- also
    Rate-Limit und die Gateway-/Timeout-Fehler, die WDQS unter Last liefert
    (nginx 502/504, interner 500). `Retry-After` wird respektiert, sonst
    exponentielles Backoff bis 60s. 4xx ausser 429 (z.B. 400 bei zu grossem
    Batch) sind echte Fehler und brechen ab.
    """
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/sparql-results+json",
    }
    params = {"query": query, "format": "json"}

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(
                WDQS_ENDPOINT, params=params, headers=headers, timeout=90
            )
        except requests.exceptions.RequestException as e:
            wait = min(2 ** attempt, 60)
            log.warning("Netzwerkfehler (%s), warte %ss...", e, wait)
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code in RETRYABLE_STATUS:
            retry_after = resp.headers.get("Retry-After")
            wait = int(retry_after) if retry_after else min(2 ** attempt, 60)
            log.warning(
                "HTTP %s vom Endpoint, warte %ss (Versuch %s/%s)...",
                resp.status_code, wait, attempt, max_retries,
            )
            time.sleep(wait)
            continue

        # Andere Fehler (z.B. 400 Bad Request bei zu grossem Batch)
        log.error("HTTP %s: %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()

    raise RuntimeError(f"Query nach {max_retries} Versuchen fehlgeschlagen.")


def qid_from_uri(uri: str) -> str:
    """Extrahiert 'Q42' aus 'http://www.wikidata.org/entity/Q42'."""
    return uri.rsplit("/", 1)[-1]


def process_batch(session: requests.Session, batch: list[str],
                   user_agent: str, langs: str) -> dict[str, dict]:
    """Fragt einen Batch ab und liefert {qid: {label, dewiki, enwiki}}."""
    query = build_query(batch, langs)
    data = run_query(session, query, user_agent)

    results: dict[str, dict] = {}
    for row in data["results"]["bindings"]:
        qid = qid_from_uri(row["item"]["value"])
        entry = results.setdefault(
            qid, {"label": None, "dewiki": None, "enwiki": None}
        )
        if "itemLabel" in row:
            entry["label"] = row["itemLabel"]["value"]
        if "dewikiTitle" in row:
            entry["dewiki"] = row["dewikiTitle"]["value"]
        if "enwikiTitle" in row:
            entry["enwiki"] = row["enwikiTitle"]["value"]

    # QIDs ohne jegliches Ergebnis (z.B. geloescht/redirected) trotzdem vermerken
    for qid in batch:
        results.setdefault(qid, {"label": None, "dewiki": None, "enwiki": None})

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Textdatei mit einer QID pro Zeile")
    parser.add_argument("output", help="Ziel-CSV-Datei")
    parser.add_argument(
        "--user-agent", required=True,
        help='Aussagekraeftiger User-Agent, z.B. "MeinProjekt/1.0 (kontakt@example.org)"',
    )
    parser.add_argument("--batch-size", type=int, default=800,
                         help="QIDs pro SPARQL-Query (Default: 800)")
    parser.add_argument("--langs", default="de,en",
                         help="Sprachprioritaet fuer Labels (Default: de,en)")
    parser.add_argument("--sleep", type=float, default=1.0,
                         help="Pause zwischen Batches in Sekunden (Default: 1.0)")
    args = parser.parse_args()

    all_qids = read_qids(args.input)
    done = load_already_done(args.output)
    todo = [q for q in all_qids if q not in done]

    log.info("Gesamt: %d QIDs, bereits erledigt: %d, offen: %d",
              len(all_qids), len(done), len(todo))

    if not todo:
        log.info("Nichts zu tun. Fertig.")
        return

    write_header = not Path(args.output).exists()
    session = requests.Session()

    with open(args.output, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["qid", "label", "dewiki_title", "enwiki_title"])
            f.flush()

        batches = [
            todo[i:i + args.batch_size]
            for i in range(0, len(todo), args.batch_size)
        ]

        for idx, batch in enumerate(batches, start=1):
            log.info("Batch %d/%d (%d QIDs)...", idx, len(batches), len(batch))
            try:
                results = process_batch(session, batch, args.user_agent, args.langs)
            except Exception as e:
                log.error("Batch %d endgueltig fehlgeschlagen: %s", idx, e)
                log.error("Abbruch. Bereits gespeicherte QIDs bleiben erhalten, "
                          "einfach das Skript erneut starten (Resume).")
                sys.exit(1)

            for qid in batch:
                r = results[qid]
                writer.writerow([qid, r["label"], r["dewiki"], r["enwiki"]])
            f.flush()

            time.sleep(args.sleep)

    log.info("Fertig. Ergebnisse in %s", args.output)


if __name__ == "__main__":
    main()