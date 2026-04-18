"""
scrape_alvaro_campos.py
-----------------------
Faz scraping de todos os poemas de Alvaro de Campos do Arquivo Pessoa
(http://arquivopessoa.net) e salva em dois formatos:

  data/poemas_alvaro_campos.json   -> lista de dicionarios com metadados
  data/dataset_alvaro_campos.csv   -> formato pronto para fine-tuning (coluna "texto")

O JSON contem todos os poemas coletados. O CSV contem apenas os poemas
em portugues, filtrados por deteccao de idioma via langdetect.

Uso:
    pip install requests beautifulsoup4 langdetect
    python src/scrape_alvaro_campos.py

Correcoes aplicadas:
  - Encoding: usa r.content + BeautifulSoup detecta charset do HTML (corrige A->A com acento)
  - Seletor de autor: usa div.autor em vez de soup.find(string=...) que nao
    encontrava texto dentro de tags
  - Extracao do titulo: primeiro <p> de div.texto-poesia
  - Extracao do corpo: demais <p> de div.texto-poesia
  - Saida em CSV compativel com finetune.py (coluna "texto")
"""

import csv
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException

# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------
BASE_URL   = "http://arquivopessoa.net"
AUTOR_ALVO = "Alvaro de Campos"

ID_START = 1
ID_END   = 4500       # margem generosa; o script pula paginas vazias ou de outros autores

DELAY_ENTRE_REQUESTS = 1.0   # segundos
OUTPUT_DIR = Path("data")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; scraper-academico/1.0; "
        "projeto-fiap; contato: rabramo@gmail.com)"
    )
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fetch(url: str):
    """Faz GET e retorna BeautifulSoup, ou None em caso de erro."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        # Usa r.content (bytes) para que o BeautifulSoup detecte o charset
        # do meta-tag do HTML corretamente
        return BeautifulSoup(r.content, "html.parser")
    except requests.RequestException as e:
        print(f"  Erro em {url}: {e}")
        return None


def extrair_poema(soup: BeautifulSoup, url: str):
    """
    Extrai titulo, autor, corpo do poema e fonte bibliografica.
    Retorna None se a pagina nao for de Alvaro de Campos.
    """
    # Verifica se e do Alvaro de Campos usando div.autor
    autor_div = soup.find("div", class_="autor")
    if not autor_div:
        return None
    if "lvaro de Campos" not in autor_div.get_text():
        return None

    # Conteudo do poema esta em div.texto-poesia
    texto_div = soup.find("div", class_="texto-poesia")
    if not texto_div:
        return None

    paragrafos = [p.get_text(strip=True) for p in texto_div.find_all("p")]
    paragrafos = [p for p in paragrafos if p]  # remove vazios

    if not paragrafos:
        return None

    # Primeiro paragrafo e o titulo; o restante e o corpo
    titulo = paragrafos[0]
    corpo  = "\n".join(paragrafos[1:]).strip()

    if not corpo:
        return None

    # Fonte bibliografica
    biblio_div = soup.find("div", class_="biblio")
    fonte = biblio_div.get_text(strip=True) if biblio_div else ""

    # Data
    data_div = soup.find("div", class_="data")
    data = data_div.get_text(strip=True) if data_div else ""

    return {
        "titulo": titulo,
        "autor":  "Alvaro de Campos",
        "data":   data,
        "url":    url,
        "corpo":  corpo,
        "fonte":  fonte,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    poemas = []
    print(f"Iniciando scraping de ID {ID_START} a {ID_END}...\n")

    for id_ in range(ID_START, ID_END + 1):
        url = f"{BASE_URL}/textos/{id_}"

        soup = fetch(url)
        if soup is None:
            if id_ % 100 == 0:
                print(f"  ... verificando id {id_}")
            continue

        poema = extrair_poema(soup, url)
        if poema:
            poemas.append(poema)
            print(f"  [{id_:04d}] {poema['titulo'][:60]}")
        else:
            if id_ % 100 == 0:
                print(f"  ... verificando id {id_}")

        time.sleep(DELAY_ENTRE_REQUESTS)

    print(f"\nTotal de poemas coletados: {len(poemas)}")

    # --- JSON completo com metadados ---
    json_path = OUTPUT_DIR / "poemas_alvaro_campos.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(poemas, f, ensure_ascii=False, indent=2)
    print(f"JSON salvo em: {json_path}")

    # --- CSV compativel com finetune.py (coluna "texto") ---
    # Apenas poemas em portugues; poemas em outro idioma ficam somente no JSON.
    csv_path = OUTPUT_DIR / "dataset_alvaro_campos.csv"
    ignorados = 0
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["texto"])
        writer.writeheader()
        for poema in poemas:
            texto = f"{poema['titulo']}\n\n{poema['corpo']}"
            try:
                idioma = detect(poema["corpo"])
            except LangDetectException:
                ignorados += 1
                continue
            if idioma != "pt":
                ignorados += 1
                continue
            writer.writerow({"texto": texto})
    print(f"CSV salvo em: {csv_path}")
    if ignorados:
        print(f"  {ignorados} poema(s) em outro idioma mantidos apenas no JSON.")

    print("\nScraping concluido!")


if __name__ == "__main__":
    main()
