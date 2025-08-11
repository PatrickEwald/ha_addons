import os
import imaplib
import email
from email.header import decode_header
from html import unescape
import json
import sys
from datetime import datetime, timedelta, timezone

# ==== OPTIONALE .env-UNTERSTÜTZUNG ====
try:
    # Falls python-dotenv installiert ist, werden lokale Umgebungsvariablen aus einer .env geladen
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

 # ==== KONFIGURATION ====
# Bevorzugt Umgebungsvariablen nutzen; fällt zurück auf sinnvolle Defaults
IMAP_SERVER = os.getenv("IMAP_SERVER", "")
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASS = os.getenv("EMAIL_PASS", "")
SUCHWÖRTER = ["Zustellung heute", "In Zustellung"]  # Suchbegriffe
ANZAHL_MAILS = int(os.getenv("ANZAHL_MAILS", "20"))  # Anzahl der letzten Mails, die geprüft werden sollen

# Debug-Flag
DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "True")

# Output JSON path for results
OUTPUT_JSON = os.getenv("OUTPUT_JSON", "/share/amazon_status.json")

# Optional: Presets für gängige Provider via IMAP_PRESET ("gmail", "gmx", "ionos", "outlook")
PRESET_SERVER = {
    "gmail": "imap.gmail.com",
    "gmx": "imap.gmx.net",
    "ionos": "imap.ionos.de",
    "outlook": "outlook.office365.com",
}
IMAP_PRESET = os.getenv("IMAP_PRESET")
if IMAP_PRESET and IMAP_PRESET.lower() in PRESET_SERVER:
    IMAP_SERVER = PRESET_SERVER[IMAP_PRESET.lower()]

# ==== FUNKTIONEN ====

def decode_mime_words(s: str) -> str:
    if not s:
        return ""
    decoded = ''
    for word, charset in decode_header(s):
        if isinstance(word, bytes):
            try:
                decoded += word.decode(charset or 'utf-8', errors='ignore')
            except Exception:
                decoded += word.decode('utf-8', errors='ignore')
        else:
            decoded += word
    return decoded


def suche_in_betreff(betreff: str, suchwoerter) -> bool:
    betreff_lower = betreff.lower()
    for wort in suchwoerter:
        if wort.lower() in betreff_lower:
            return True
    return False




# ==== HAUPTPROGRAMM ====

def main():
    if not EMAIL_USER or not EMAIL_PASS:
        raise SystemExit("Bitte EMAIL_USER und EMAIL_PASS als Umgebungsvariablen setzen oder im Skript konfigurieren.")

    IMAGE_SAVE_DIR = os.getenv("IMAGE_SAVE_DIR", "./images")
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

    # Verbindung herstellen
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("INBOX")
    if DEBUG:
        print(f"[DEBUG] IMAP_SERVER={IMAP_SERVER} EMAIL_USER={EMAIL_USER}")

    # Letzte Mails suchen (nur die letzten 3 Tage, bis jetzt)
    now = datetime.now(timezone.utc)
    since_date = (now - timedelta(days=3)).date()  # inkl. heutigem Tag
    before_date = (now + timedelta(days=1)).date()  # IMAP BEFORE ist exklusiv; +1 Tag, um heute einzuschließen

    since_str = since_date.strftime("%d-%b-%Y")
    before_str = before_date.strftime("%d-%b-%Y")

    # Kombiniere SINCE und BEFORE, damit der Zeitraum exakt begrenzt ist
    status, daten = mail.search(None, 'SINCE', since_str, 'BEFORE', before_str)
    if status != 'OK' or not daten or not daten[0]:
        print("Keine Mails im definierten Zeitraum gefunden (letzte 3 Tage).")
        mail.logout()
        return

    mail_ids = daten[0].split()
    if DEBUG:
        print(f"[DEBUG] Zeitraum: SINCE {since_str} AND BEFORE {before_str}")
        print(f"[DEBUG] Treffer im Zeitraum: {len(mail_ids)}")

    # Optional weiter begrenzen, um nicht zu viele Mails zu verarbeiten
    letzte_ids = mail_ids[-ANZAHL_MAILS:] if ANZAHL_MAILS > 0 else mail_ids
    if DEBUG:
        print(f"[DEBUG] Prüfe die letzten {len(letzte_ids)} Mails im Zeitraum")

    results = {
        "zustellung_heute": [],
        "zugestellt": []
    }

    import re
    # Robustere Muster
    re_delivered = re.compile(r"\bzugestellt\b.*?(\d{3}-\d{7}-\d{7})|Bestellnummer\s*(\d{3}-\d{7}-\d{7})", re.IGNORECASE)
    re_today_phrase = re.compile(r"\b(?:zustellung|lieferung|ankunft|kommt|kommt\s+an)\s*(?:voraussichtlich|geplant)?\s*heute\b", re.IGNORECASE)
    re_in_zustellung_subject = re.compile(r"\bIn\s+Zustellung\b", re.IGNORECASE)
    # klassische Variante mit Trennzeichen
    re_time_window = re.compile(
        r"(?:heute|zustellung\s*heute|lieferung\s*heute|ankunft\s*heute)?[^\n\r]{0,80}?"  # optionaler Prefix nahe dran
        r"(\b\d{1,2}:\d{2}\b)\s*(?:–|-|—|bis|und|–|-)\s*(\b\d{1,2}:\d{2}\b)",
        re.IGNORECASE,
    )
    # 'heute' irgendwo vor den Zeiten (robuster)
    re_today_window_relaxed = re.compile(
        r"heute[^\n\r]{0,120}?(\b\d{1,2}:\d{2}\b)[^\n\r]{0,30}?(?:–|-|—|bis|und)\s*(\b\d{1,2}:\d{2}\b)",
        re.IGNORECASE,
    )
    # 'zwischen HH:MM und HH:MM' (ggf. mit 'heute' davor)
    re_between_window = re.compile(
        r"(?:heute[^\n\r]{0,40})?zwischen\s*(\b\d{1,2}:\d{2}\b)\s*(?:und|bis)\s*(\b\d{1,2}:\d{2}\b)",
        re.IGNORECASE,
    )
    re_order_body = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")
    re_order_fallback = re.compile(r"Bestell(?:nr\.|nummer)\s*[:#]?\s*(\d{3}-\d{7}-\d{7}|[A-Z0-9-]{6,})", re.IGNORECASE)

    for mail_id in reversed(letzte_ids):
        
        status, daten = mail.fetch(mail_id, "(RFC822.HEADER)")
        if status != 'OK' or not daten or not isinstance(daten[0], tuple):
            continue
        raw_header = daten[0][1]
        msg = email.message_from_bytes(raw_header)

        # Betreff & From decodieren
        subject = decode_mime_words(msg.get("Subject", ""))
        from_ = decode_mime_words(msg.get("From", ""))

        date_raw = msg.get("Date", "")
        try:
            from email.utils import parsedate_to_datetime
            date_parsed = parsedate_to_datetime(date_raw)
            if date_parsed:
                date_str = date_parsed.strftime("%Y-%m-%d %H:%M:%S")
            else:
                date_str = date_raw
        except Exception:
            date_str = date_raw

        # Zusätzlicher Filter: "Zugestellt: Bestellnummer <ORDER>" im Betreff erkennen
        m_deliv = re_delivered.search(subject)
        delivered_match = None
        if m_deliv:
            delivered_match = next((g for g in m_deliv.groups() if g), None)
        # Subjekt-Hinweise
        subject_has_today = bool(re_today_phrase.search(subject) or re_in_zustellung_subject.search(subject))

        # ==== Body immer prüfen, wenn wir irgendeinen Hinweis haben ODER generell, um Fälle wie im Screenshot abzudecken ====
        status_full, daten_full = mail.fetch(mail_id, "(RFC822)")
        order_from_subject = delivered_match if delivered_match else None
        order_from_body = None
        time_window = None
        is_zustellung_heute = subject_has_today

        if status_full == 'OK' and daten_full and isinstance(daten_full[0], tuple):
            full_msg = email.message_from_bytes(daten_full[0][1])
            body_texts = []
            if full_msg.is_multipart():
                for part in full_msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition", ""))
                    if "attachment" in content_disposition:
                        continue
                    try:
                        charset = part.get_content_charset() or "utf-8"
                    except Exception:
                        charset = "utf-8"
                    payload = part.get_payload(decode=True)
                    if not payload:
                        continue
                    try:
                        text = payload.decode(charset, errors="ignore")
                    except Exception:
                        text = payload.decode("utf-8", errors="ignore")
                    if content_type == "text/plain":
                        body_texts.append(text)
                    elif content_type == "text/html":
                        import re as _re
                        html_raw = text
                        text = unescape(_re.sub(r"<[^>]+>", " ", html_raw))
                        body_texts.append(text)
            else:
                try:
                    charset = full_msg.get_content_charset() or "utf-8"
                except Exception:
                    charset = "utf-8"
                payload = full_msg.get_payload(decode=True)
                if payload:
                    try:
                        text = payload.decode(charset, errors="ignore")
                    except Exception:
                        text = payload.decode("utf-8", errors="ignore")
                    if full_msg.get_content_type() == "text/html":
                        import re as _re
                        html_raw = text
                        text = unescape(_re.sub(r"<[^>]+>", " ", html_raw))
                    body_texts.append(text)

            body_combined = "\n".join(body_texts)

            # Falls "Zustellung heute" nur im Body steht bzw. Synonyme
            if not is_zustellung_heute and (re_today_phrase.search(body_combined) or re_today_window_relaxed.search(body_combined) or re_between_window.search(body_combined)):
                is_zustellung_heute = True

            # Zeitfenster ermitteln: mehrere Muster versuchen
            m_time = (
                re_time_window.search(body_combined)
                or re_between_window.search(body_combined)
                or re_today_window_relaxed.search(body_combined)
            )
            if m_time:
                time_window = f"{m_time.group(1)} - {m_time.group(2)}"

            if DEBUG and not (m_time or delivered_match or subject_has_today):
                excerpt = body_combined[:300].replace("\n", " ")
                print(f"[DEBUG] Kein Zeitfenster erkannt. Body-Auszug: {excerpt}")

            # Bestellnummer aus Body
            m_order = re_order_body.search(body_combined)
            if not m_order:
                m_order = re_order_fallback.search(body_combined)
            if m_order:
                order_from_body = m_order.group(1) if m_order.lastindex else m_order.group(0)


        # Rekord bauen NUR wenn eine Kategorie zutrifft
        if is_zustellung_heute:
            record = {
                "subject": subject,
                "order_number": order_from_subject or order_from_body,
                "time_window": time_window,
                "date": date_str,
            }
            if DEBUG:
                print(f"[DEBUG] ZustellungHeute: subject='{subject}' order={record['order_number']} time={record['time_window']}")
            results["zustellung_heute"].append(record)

        if delivered_match:
            record = {
                "subject": subject,
                "order_number": order_from_subject or order_from_body,
                "date": date_str,
            }
            results["zugestellt"].append(record)

    if DEBUG and not results["zustellung_heute"] and not results["zugestellt"]:
        print("[DEBUG] Keine Treffer. Letzte Betreffe zur Kontrolle:")
        # Fetch again lightweight subjects for the inspected IDs
        for mail_id in reversed(letzte_ids):
            status, daten = mail.fetch(mail_id, "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM)])")
            if status != 'OK' or not daten or not isinstance(daten[0], tuple):
                continue
            hdr = email.message_from_bytes(daten[0][1])
            s = decode_mime_words(hdr.get("Subject", ""))
            f = decode_mime_words(hdr.get("From", ""))
            print(f"[DEBUG] From={f} | Subject={s}")

    # ---- Nachbearbeitung: bereits zugestellte Bestellungen aus "zustellung_heute" entfernen + Duplikate bereinigen ----
    try:
        # Set aller zugestellten Bestellnummern (nur mit vorhandener order_number)
        delivered_orders = {
            r.get("order_number") for r in results.get("zugestellt", []) if r.get("order_number")
        }

        if delivered_orders:
            before_len = len(results.get("zustellung_heute", []))
            results["zustellung_heute"] = [
                r for r in results.get("zustellung_heute", [])
                if r.get("order_number") not in delivered_orders
            ]
            if DEBUG:
                print(f"[DEBUG] ZustellungHeute: {before_len - len(results['zustellung_heute'])} Einträge entfernt (bereits zugestellt)")

        def _dedupe(records, key_fields):
            seen = set()
            out = []
            for rec in records:
                key = tuple(rec.get(k) for k in key_fields)
                if key in seen:
                    continue
                seen.add(key)
                out.append(rec)
            return out

        # Duplikate entfernen (auf sinnvollen Schlüsseln)
        results["zustellung_heute"] = _dedupe(
            results.get("zustellung_heute", []),
            ["order_number", "subject", "date", "time_window"],
        )
        results["zugestellt"] = _dedupe(
            results.get("zugestellt", []),
            ["order_number", "subject", "date"],
        )
    except Exception as e:
        print(f"[WARN] Nachbearbeitung fehlgeschlagen: {e}")

    # ---- Ergebnisse auch als Datei ablegen, damit Home Assistant sie lesen kann ----
    try:
        # atomar schreiben: zuerst temp-Datei, dann umbenennen
        _tmp_path = OUTPUT_JSON + ".tmp"
        with open(_tmp_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        os.replace(_tmp_path, OUTPUT_JSON)
        if DEBUG:
            print(f"[DEBUG] Ergebnisse nach {OUTPUT_JSON} geschrieben")
    except Exception as e:
        print(f"[WARN] Konnte Ergebnisse nicht nach {OUTPUT_JSON} schreiben: {e}")

    # Für Log/Debug weiterhin auf STDOUT ausgeben
    print(json.dumps(results, ensure_ascii=False, indent=2))
    mail.logout()


if __name__ == "__main__":
    main()