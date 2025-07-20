import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import json
import random
from pathlib import Path
from config import settings
import sys

AVAILABLE_ROLES = settings.DOC_ROLES

URL_FILE = "scripts/urls.txt"
OUTPUT_DIR = Path("data/parsed_docs")
OUTPUT_DIR.mkdir(exist_ok=True)

def fetch_and_parse(url: str) -> dict:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"It was not possible to load {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.string.strip() if soup.title else "No Title"
    content_div = (
            soup.find("div", class_="wiki-content") or
            soup.find("div", id="main-content") or
            soup.find("div", attrs={"data-testid": "article-content"})
    )


    if not content_div:
        print(f"Content was not found {url}")
        return None

    markdown = md(str(content_div))
    role = random.choice(AVAILABLE_ROLES)

    return {
        "title": title,
        "content": markdown.strip(),
        "source_url": url,
        "role": role
    }

def process_urls(file_path: str):
    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Processing: {url}")
        result = fetch_and_parse(url)
        if result:
            filename = OUTPUT_DIR / f"doc_{i:03}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Saved: {filename.name} (role: {result['role']})\n")

if __name__ == "__main__":
    url_file = sys.argv[1]
    if url_file is None:
        url_file = URL_FILE
    process_urls(url_file)
