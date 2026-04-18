"""
json_to_csv.py
--------------
Gera data/dataset_alvaro_campos.csv a partir do data/poemas_alvaro_campos.json
ja existente, filtrando apenas os poemas em portugues.

Util para nao precisar re-executar o scraper quando o JSON ja foi coletado.

Uso:
    pip install langdetect
    python src/json_to_csv.py
"""

import csv
import json
from pathlib import Path

from langdetect import detect, LangDetectException

BASE_DIR  = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "data" / "poemas_alvaro_campos.json"
CSV_PATH  = BASE_DIR / "data" / "dataset_alvaro_campos.csv"


def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON nao encontrado em: {JSON_PATH}")

    with open(JSON_PATH, encoding="utf-8") as f:
        poemas = json.load(f)

    print(f"Total no JSON: {len(poemas)} poemas")

    incluidos = 0
    ignorados = 0

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["texto"])
        writer.writeheader()

        for poema in poemas:
            corpo = poema.get("corpo", "").strip()
            if not corpo:
                ignorados += 1
                continue

            try:
                idioma = detect(corpo)
            except LangDetectException:
                ignorados += 1
                continue

            if idioma != "pt":
                print(f"  Ignorado ({idioma}): {poema.get('titulo', '')[:60]}")
                ignorados += 1
                continue

            texto = f"{poema['titulo']}\n\n{corpo}"
            writer.writerow({"texto": texto})
            incluidos += 1

    print(f"\nCSV gerado em: {CSV_PATH}")
    print(f"  Incluidos (pt): {incluidos}")
    print(f"  Ignorados     : {ignorados}")


if __name__ == "__main__":
    main()
