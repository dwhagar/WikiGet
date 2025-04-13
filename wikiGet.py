import os
import re
import json
import requests
import argparse
from datetime import datetime
from rich.console import Console
from rich.progress import Progress
from concurrent.futures import ProcessPoolExecutor

# Console and constants
console = Console(log_time=True, log_path=False)

WIKI_URL = "https://wiki.moltenaether.com/w/api.php"
API_KEY = None
CATEGORY_BLACKLIST = [
    "Abandoned", "OOC", "Sources of Inspiration", "Star Trek", "Star Wars",
    "Talk Pages", "Denied or Banned", "Unapproved"
]
PAGE_BLACKLIST = ["User:", "Talk:", "Template:", "Special:"]
LLM_MODEL = "all-MiniLM-L6-v2"
LLM_VECTOR_ROUND = 3
max_workers = round(os.cpu_count() * 0.83)
visited = set()
queue = []
total_pages = 0

# Load models
console.log("[blue]Loading embedding model (all-MiniLM-L6-v2)...")
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
EMBED_MODEL = SentenceTransformer(LLM_MODEL)
KEYBERT_MODEL = KeyBERT(model=EMBED_MODEL)

# --- Utilities ---

def sanitize_filename(title: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', ' ', title)[:255]

def build_page_url(title: str) -> str:
    import urllib.parse
    return f"{WIKI_URL.replace('/api.php', '')}/index.php?title=" + urllib.parse.quote(title.replace(" ", "_"))

# --- Parser ---

def parse_wiki_text_into_summary_and_sections(text: str) -> dict:
    """
    Parses wikitext into summary and section structure:
    - summary = content before first heading
    - sections = level-2 (==) headings
    - subsections = level-3 (===) headings within a section
    """
    heading_regex = re.compile(r"^(={2,3})\s*(.*?)\s*\1\s*$", re.MULTILINE)
    result = {"summary": "", "sections": []}
    matches = list(heading_regex.finditer(text))
    last_pos = 0

    if matches:
        result["summary"] = text[:matches[0].start()].strip()
        last_pos = matches[0].start()
    else:
        result["summary"] = text.strip()
        return result

    current_section = None
    current_subsection = None

    for match in matches:
        level = len(match.group(1))
        heading = match.group(2).strip()
        section_text = text[last_pos:match.start()]
        last_pos = match.end()

        if level == 2:
            # Finalize any open subsection
            if current_subsection:
                if current_section is None:
                    current_section = {
                        "heading": "Unnamed Section",
                        "content": "",
                        "subsections": []
                    }
                current_subsection["content"] = current_subsection["content"].strip()
                current_section["subsections"].append(current_subsection)
                current_subsection = None

            # Finalize current section
            if current_section:
                current_section["content"] = current_section["content"].strip()
                result["sections"].append(current_section)

            # Start new section
            current_section = {
                "heading": heading,
                "content": "",
                "subsections": []
            }

        elif level == 3:
            # Finalize previous subsection
            if current_subsection:
                if current_section is None:
                    current_section = {
                        "heading": "Unnamed Section",
                        "content": "",
                        "subsections": []
                    }
                current_subsection["content"] = current_subsection["content"].strip()
                current_section["subsections"].append(current_subsection)

            # Create new subsection
            current_subsection = {
                "heading": heading,
                "content": ""
            }

        # Append content to correct location
        if current_subsection:
            current_subsection["content"] += section_text
        elif current_section:
            current_section["content"] += section_text
        else:
            result["summary"] += "\n" + section_text  # fallback

    # Final trailing text
    trailing_text = text[last_pos:].strip()
    if current_subsection:
        current_subsection["content"] += trailing_text
        if current_section is None:
            current_section = {
                "heading": "Unnamed Section",
                "content": "",
                "subsections": []
            }
        current_section["subsections"].append(current_subsection)
    elif current_section:
        current_section["content"] += trailing_text
    else:
        result["summary"] += "\n" + trailing_text

    # Final append
    if current_section:
        current_section["content"] = current_section["content"].strip()
        result["sections"].append(current_section)

    # Tidy subsection content
    for sec in result["sections"]:
        for sub in sec["subsections"]:
            sub["content"] = sub["content"].strip()

    result["summary"] = result["summary"].strip()
    return result

# --- Fetch ---

def fetch_all_pages() -> list:
    global total_pages
    pages = []
    apcontinue = None
    params = {
        "action": "query", "list": "allpages", "format": "json",
        "aplimit": "max", "apfilterredir": "nonredirects"
    }
    if API_KEY: params["apikey"] = API_KEY

    while True:
        if apcontinue:
            params["apcontinue"] = apcontinue
        else:
            params.pop("apcontinue", None)
        try:
            response = requests.get(WIKI_URL, params=params, timeout=30).json()
        except:
            break
        pages.extend(p["title"] for p in response["query"]["allpages"])
        apcontinue = response.get("continue", {}).get("apcontinue")
        if not apcontinue:
            break

    total_pages = len(pages)
    return pages

def fetch_wiki_name() -> str:
    try:
        res = requests.get(WIKI_URL, params={
            "action": "query", "meta": "siteinfo", "format": "json", "siprop": "general"
        }, timeout=30).json()
        return res["query"]["general"]["sitename"]
    except:
        return "UnknownWiki"

def fetch_page_data(title: str) -> dict:
    if any(prefix.lower() in title.lower() for prefix in PAGE_BLACKLIST):
        return {"title": title, "blacklisted": True}

    params = {
        "action": "query", "titles": title, "prop": "revisions|categories|links",
        "rvprop": "content", "cllimit": "max", "pllimit": "max", "format": "json"
    }
    if API_KEY:
        params["apikey"] = API_KEY

    try:
        response = requests.get(WIKI_URL, params=params, timeout=30).json()
    except:
        return {"title": title, "fetch_error": True}

    pages = response.get("query", {}).get("pages", {})
    for _, page in pages.items():
        raw = page.get("revisions", [{}])[0].get("*", "")
        parsed = parse_wiki_text_into_summary_and_sections(raw)
        cats = [c["title"] for c in page.get("categories", [])]
        if any(bc.lower() in c.lower() for bc in CATEGORY_BLACKLIST for c in cats):
            return {"title": title, "blacklisted": True}
        links = [l["title"] for l in page.get("links", []) if not any(p in l["title"] for p in PAGE_BLACKLIST)]
        return {
            "title": title,
            "url": build_page_url(title),
            "summary": parsed["summary"],
            "sections": parsed["sections"],
            "categories": cats,
            "links": links,
            "raw_text": raw
        }

    return {"title": title, "fetch_error": True}

# --- Phase 1: Crawl ---

def crawl_wiki(pages: list, verbose=False) -> list:
    output = []
    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Processing pages...", total=len(pages), highlight=False)
        for title in pages:
            data = fetch_page_data(title)
            if verbose:
                if data.get("blacklisted"):
                    progress.log(f"[yellow]Blacklisted:[/] {title}", highlight=False)
                elif data.get("fetch_error"):
                    progress.log(f"[red]Fetch error:[/] {title}", highlight=False)
                else:
                    progress.log(f"[green]Fetched:[/] {title}", highlight=False)
            if not data.get("blacklisted") and not data.get("fetch_error"):
                output.append(data)
            visited.add(title)
            progress.update(task, advance=1)
    return output

# --- Phase 2: Enrich ---

def generate_embedding(text: str, decimals=3) -> list:
    raw_vector = EMBED_MODEL.encode(text)
    # Convert each float in `raw_vector` to a rounded float
    return [round(float(v), decimals) for v in raw_vector]

def extract_keywords(text: str) -> list:
    keywords = KEYBERT_MODEL.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words="english", use_maxsum=True, top_n=10)
    return [kw for kw, _ in keywords]

def enrich_page(page: dict) -> dict:
    page["summary_embedding"] = generate_embedding(page["summary"], decimals=LLM_VECTOR_ROUND)
    for section in page["sections"]:
        section["embedding"] = generate_embedding(section["content"], decimals=LLM_VECTOR_ROUND)
        for subsection in section.get("subsections", []):
            subsection["embedding"] = generate_embedding(subsection["content"], decimals=LLM_VECTOR_ROUND)
    page["keywords"] = extract_keywords(page["raw_text"])
    del page["raw_text"]
    return page

def enrich_pages(pages: list, workers: int = None, verbose=False) -> list:
    enriched = []
    with Progress(console=console) as progress:
        task = progress.add_task("[magenta]Generating embeddings...", total=len(pages))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(enrich_page, pages):
                if verbose:
                    progress.log(f"[green]Enriched:[/] {result['title']}", highlight=False)
                enriched.append(result)
                progress.update(task, advance=1)
    return enriched

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-w", "--wiki-name", help="Override wiki name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    wiki_name = args.wiki_name or fetch_wiki_name()
    safe_name = sanitize_filename(wiki_name)
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_file = args.output or f"{safe_name} Data {date_str}.json"

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        console.log(f"[red]Output folder does not exist:[/] {output_dir}", highlight=False)
        exit(1)

    console.log("[green]Fetching page list...", highlight=False)
    all_pages = fetch_all_pages()

    console.log("[green]Starting crawl...", highlight=False)
    raw_data = crawl_wiki(all_pages, verbose=args.verbose)
    console.log("[green]Starting enrichment...", highlight=False)
    enriched_data = enrich_pages(raw_data, workers=max_workers, verbose=args.verbose)

    output = {
        "wiki_name": wiki_name,
        "content_format": "wikitext",
        "embedding_model": LLM_MODEL,
        "keyword_model": "keybert with " + LLM_MODEL,
        "pages": enriched_data,
        "categories": {}
    }

    for page in enriched_data:
        for cat in page["categories"]:
            output["categories"].setdefault(cat, []).append(page["title"])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    console.log(f"[bold green]Data written to:[/] {output_file}", highlight=False)

if __name__ == "__main__":
    main()
