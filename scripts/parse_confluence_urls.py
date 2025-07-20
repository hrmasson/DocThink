import json
import logging
import random
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from application.config import settings

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

AVAILABLE_ROLES = settings.DOC_ROLES
DEFAULT_URL_FILE = "scripts/urls.txt"
OUTPUT_DIR = Path("data/parsed_docs")
OUTPUT_DIR.mkdir(exist_ok=True)


def fetch_and_parse(url: str) -> dict | None:
    """Fetch HTML content from a URL and extract relevant article info."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.string.strip() if soup.title else "No Title"
    content_div = (
            soup.find("div", class_="wiki-content")
            or soup.find("div", id="main-content")
            or soup.find("div", attrs={"data-testid": "article-content"})
    )

    if not content_div:
        logger.warning(f"No main content found at: {url}")
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
    """Process all URLs listed in the given text file."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"URL file not found: {file_path}")
        return

    with path.open("r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        logger.warning("No URLs found in the input file.")
        return

    logger.info(f"Processing {len(urls)} URLs...")

    for i, url in enumerate(urls, 1):
        logger.info(f"[{i}/{len(urls)}] Fetching: {url}")
        result = fetch_and_parse(url)
        if result:
            filename = OUTPUT_DIR / f"doc_{i:03}.json"
            try:
                with filename.open("w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved: {filename.name} (role: {result['role']})")
            except Exception as e:
                logger.error(f"Failed to save {filename.name}: {e}")


if __name__ == "__main__":
    url_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL_FILE
    process_urls(url_file)
