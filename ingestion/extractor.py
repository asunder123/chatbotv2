# ingestion/extractor.py
# File upload → raw text extraction

import re
import json
import zipfile

def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith((".txt", ".md", ".csv")):
        return raw.decode("utf-8", errors="ignore")

    if name.endswith(".json"):
        try:
            return json.dumps(json.loads(raw.decode("utf-8", errors="ignore")), indent=2)
        except Exception:
            return raw.decode("utf-8", errors="ignore")

    if name.endswith(".html"):
        return re.sub(r"<[^>]+>", " ", raw.decode("utf-8", errors="ignore"))

    if name.endswith(".docx"):
        with zipfile.ZipFile(uploaded_file) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
            return re.sub(r"<[^>]+>", " ", xml)

    if name.endswith(".pdf"):
        text = raw.decode("latin1", errors="ignore")
        return re.sub(r"[^A-Za-z0-9.,;:?!()\n ]+", " ", text)

    return raw.decode("utf-8", errors="ignore")
