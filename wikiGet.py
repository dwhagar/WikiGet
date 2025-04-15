import os
import re
import json
import requests
import argparse
import numpy as np
import signal
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, track
from multiprocessing import Pool
from collections import defaultdict

# Global Variables
WIKI_URL = "https://wiki.moltenaether.com/w/api.php"
API_KEY = None # Usually you won't need an API key to read

# Blacklists are found anywhere in the page name or category name
CATEGORY_BLACKLIST = [
    "Abandoned", "OOC", "Sources of Inspiration", "Star Trek", "Star Wars",
    "Talk Pages", "Denied or Banned", "Unapproved"
]
PAGE_BLACKLIST = ["User:", "Talk:", "Template:", "Special:"]

# The two models I recommend for this are these, you can use any model
# you want, but these should be the best to use.

# 1. "all-MiniLM-L6-v2" (recommended default)
#    - ✅ Fast and lightweight (384 dimensions)
#    - ⚠️ Slightly lower semantic accuracy than mpnet
#    - Ideal for large-scale or CPU-bound tasks
#
# 2. "all-mpnet-base-v2"
#    - ✅ Higher accuracy (768 dimensions)
#    - ⚠️ Slower and uses more memory
#    - Better for precision-critical applications

# These control the LLM model used for both keywords and embedding
# doing this can increase the JSON file size by 5 to 20 times, so
# implemented a rounding to 3 places to help reduce file size.
LLM_MODEL = "all-MiniLM-L6-v2"
LLM_VECTOR_ROUND = 0 # 0 to disable rounding

# Console and constants
console = Console(log_time=True, log_path=False)
max_workers = round(os.cpu_count() * 0.83)
visited = set()
queue = []
total_pages = 0

# Load models
# console.log(f"[blue]Loading embedding model ({LLM_MODEL})...")
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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

def crawl_wiki(pages: list = None, verbose: bool = False) -> dict:
    """
    Crawl wiki pages and return structured raw data.
    If no pages are passed in, fetches all non-redirect titles.

    Returns:
        dict with keys:
          - 'pages': list of fetched usable page dicts
          - 'categories': dict of category -> [titles]
          - 'wiki_name': str
    """
    if pages is None:
        pages = fetch_all_pages()

    output_pages = []
    category_map = {}
    wiki_name = fetch_wiki_name()

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

            if data.get("blacklisted") or data.get("fetch_error"):
                visited.add(title)
                progress.update(task, advance=1)
                continue

            output_pages.append(data)
            visited.add(title)

            # Update category map
            for cat in data.get("categories", []):
                category_map.setdefault(cat, []).append(title)

            progress.update(task, advance=1)

    return {
        "pages": output_pages,
        "categories": category_map,
        "wiki_name": wiki_name
    }

# --- Phase 2: Enrich ---

def compute_cosine_similarity_matrix(embeddings: list[list[float]]) -> np.ndarray:
    """
    Computes a full cosine similarity matrix for a list of embeddings.
    Returns an (n x n) matrix of float values.
    """
    embedding_matrix = np.array(embeddings)
    return cosine_similarity(embedding_matrix)


def get_top_n_similar(index: int, similarity_matrix: np.ndarray, titles: list[str], N: int) -> list[dict]:
    """
    Returns the top-N most similar items to the given index, excluding itself.
    Includes similarity scores for each.
    """
    scores = similarity_matrix[index]
    ranked_indices = np.argsort(scores)[::-1]  # descending
    results = []
    for i in ranked_indices:
        if i == index:
            continue
        results.append({
            "title": titles[i],
            "similarity": round(float(scores[i]), 3)
        })
        if len(results) >= N:
            break
    return results


def extract_all_section_embeddings(data: list[dict]) -> tuple[list[dict], list[list[float]]]:
    """
    Flattens all sections across all pages into a list, and extracts their embeddings.
    Each returned section includes:
      - `page_title`
      - `section_title`
      - `title` (combined for similarity labels)
      - `semantic_embedding`
      - `ref` to original section object (for modifying)
    """
    flat_sections = []
    embeddings = []

    for page in data:
        page_title = page["title"]
        for section in page.get("sections", []):
            section_title = section.get("heading", "Unnamed Section")
            full_label = f"{page_title} - {section_title}"

            flat_section = {
                "page_title": page_title,
                "section_title": section_title,
                "title": full_label,  # Used for related_pages
                "semantic_embedding": section["semantic_embedding"],
                "ref": section  # backref for assignment
            }

            flat_sections.append(flat_section)
            embeddings.append(section["semantic_embedding"])

    return flat_sections, embeddings


def cluster_sections(embeddings: list[list[float]], n_clusters: int = 20) -> list[int]:
    """
    Applies KMeans clustering to section embeddings.
    Returns a list of cluster IDs (same order as embeddings).
    """
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    return model.fit_predict(embeddings)


def generate_related_pages(data: list[dict], N: int = 5) -> None:
    """
    For each item in `data`, generate a `related_pages` list of top-N most similar items.
    Assumes each entry has 'semantic_embedding' and a 'title'.
    """
    embeddings = [entry["semantic_embedding"] for entry in data]
    titles = [entry["title"] for entry in data]
    sim_matrix = compute_cosine_similarity_matrix(embeddings)

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Computing related pages...", total=len(data))

        for idx, entry in enumerate(data):
            scores = sim_matrix[idx]
            ranked = np.argsort(scores)[::-1]

            related = []
            for i in ranked:
                if i == idx:
                    continue
                related.append({
                    "title": titles[i],
                    "similarity": round(float(scores[i]), 3)
                })
                if len(related) >= N:
                    break

            entry["related_pages"] = related
            progress.update(task, advance=1)


def assign_clusters_to_sections(flat_sections: list[dict], cluster_ids: list[int]) -> None:
    """
    Attaches cluster IDs to the original section references in the data structure.
    """
    for section, cluster_id in zip(flat_sections, cluster_ids):
        section["ref"]["cluster"] = int(cluster_id)


def generate_embedding(text: str, decimals=0) -> list:
    raw_vector = EMBED_MODEL.encode(text)
    if decimals < 1:
        return list(raw_vector)
    else:
        return [round(float(v), decimals) for v in raw_vector]


def extract_keywords(text: str) -> list:
    keywords = KEYBERT_MODEL.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words="english", use_maxsum=True, top_n=10)
    return [kw for kw, _ in keywords]


def enrich_page_verbose(page):
    console.log(f"[green]Enriching:[/] {page['title']}")
    return enrich_page(page)


def enrich_page(page: dict) -> dict:
    """
    Enrich a wiki page by:
    - Generating a semantic embedding for each section (including summary)
    - Generating keywords from the full raw_text
    - Removing raw_text to reduce storage
    """
    # Embed the summary as a semantic anchor
    page["semantic_embedding"] = generate_embedding(page["summary"])

    # Enrich each section with its own embedding
    for section in page.get("sections", []):
        combined_text = f"{section['heading']}\n\n{section['content']}".strip()
        section["semantic_embedding"] = generate_embedding(combined_text)
        # For easier access in flattening
        section["full_text"] = combined_text

    # Extract page-level keywords from the full raw text
    page["keywords"] = extract_keywords(page["raw_text"])

    # Clean up
    del page["raw_text"]
    return page

def generate_cluster_labels(flat_sections, cluster_ids, top_n=3):
    """
    Generate human-readable cluster names based on TF-IDF terms.
    Returns dict mapping cluster_id -> cluster_label.
    """
    cluster_texts = defaultdict(list)
    for section, cluster_id in zip(flat_sections, cluster_ids):
        heading = section.get("heading") or section["ref"].get("heading") or "Untitled"
        content = section["ref"].get("content", "")
        text = f"{heading}\n{content}"
        cluster_texts[cluster_id].append(text)

    cluster_labels = {}

    with Progress(console=console) as progress:
        task = progress.add_task("[magenta]Generating cluster labels...", total=len(cluster_texts))

        for cluster_id, texts in cluster_texts.items():
            try:
                if not texts:
                    cluster_labels[cluster_id] = f"cluster_{cluster_id}"
                    continue

                vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
                X = vectorizer.fit_transform(texts)
                terms = vectorizer.get_feature_names_out()
                tfidf_scores = X.sum(axis=0).A1
                sorted_terms = sorted(zip(terms, tfidf_scores), key=lambda x: x[1], reverse=True)
                keywords = [term for term, _ in sorted_terms[:top_n]]
                cluster_labels[cluster_id] = ", ".join(keywords)
            except Exception as e:
                cluster_labels[cluster_id] = f"cluster_{cluster_id}"
                console.log(f"[yellow]Warning: Failed to label cluster {cluster_id}: {e}")

            progress.update(task, advance=1)

    return cluster_labels

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Semantic Wiki Processor with Clustering")
    parser.add_argument("-o", "--output", help="Output .json path", default=None)
    parser.add_argument("-w", "--wiki-name", help="Override wiki name")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    args = parser.parse_args()

    # Step 1: Crawl wiki content
    raw_data = crawl_wiki(verbose=args.verbose)

    # Step 2: Enrich pages
    console.log("[blue]Enriching pages with embeddings and keywords...")

    # Set SIGINT to default so parent can interrupt children cleanly
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        with Pool(processes=args.workers or max_workers) as pool:
            worker_fn = enrich_page_verbose if args.verbose else enrich_page
            enriched_pages = list(track(
                pool.imap(worker_fn, raw_data["pages"], chunksize=1),

                total=len(raw_data["pages"]),
                description="[cyan]Enriching pages..."
            ))
    except KeyboardInterrupt:
        console.log("[red]User interrupt detected. Terminating worker pool...")
        pool.terminate()
        pool.join()
        exit(1)

    # Step 3–7: Clustering, labeling, and similarity
    flat_sections, section_embeddings = extract_all_section_embeddings(enriched_pages)
    console.log("[blue]Clustering sections...")
    section_cluster_ids = cluster_sections(section_embeddings, n_clusters=20)
    assign_clusters_to_sections(flat_sections, section_cluster_ids)

    console.log("[blue]Generating human-readable cluster labels...")
    cluster_labels = generate_cluster_labels(flat_sections, section_cluster_ids)
    for section, cid in zip(flat_sections, section_cluster_ids):
        section["ref"]["cluster_label"] = cluster_labels[cid]

    console.log("[blue]Generating related pages...")
    generate_related_pages(enriched_pages, N=5)
    console.log("[blue]Generating related sections...")
    generate_related_pages(flat_sections, N=5)

    # Step 8: Strip vectors for storage
    for page in enriched_pages:
        page.pop("semantic_embedding", None)
        for section in page.get("sections", []):
            section.pop("semantic_embedding", None)
    for section in flat_sections:
        section.pop("semantic_embedding", None)

    # Step 9: Output
    wiki_name = args.wiki_name or raw_data["wiki_name"]
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_path = args.output or f"{wiki_name} Data {date_str}.json"

    final_data = {
        "wiki_name": wiki_name,
        "content_format": "wikitext",
        "embedding_model": f"sentence-transformers/{LLM_MODEL}",
        "pages": enriched_pages,
        "sections": flat_sections,
        "categories": raw_data["categories"],
        "clusters": cluster_labels
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    console.log(f"[green]Final data saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.log("[red]Process interrupted by user. Exiting.")
        exit(1)