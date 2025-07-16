# ────────────────────────── Standard library ──────────────────────────
import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

# ───────────────────────── Third-party packages ────────────────────────
import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,   # superclass for our EMA column
)
from rich.text import Text
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
# noinspection PyUnresolvedReferences
from keybert._maxsum import max_sum_distance
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────── Rich < 10 fallback ───────────────────────────
try:
    from rich.progress import format_time  # Rich ≥ 10
except ImportError:                        # pragma: no cover
    def format_time(seconds: float | None) -> str:  # noqa: E302
        if seconds is None or seconds == float("inf"):
            return "--:--:--"
        seconds = int(seconds + 0.5)
        h, rmdr = divmod(seconds, 3600)
        m, s   = divmod(rmdr, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

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
LLM_MODEL = "all-mpnet-base-v2"
LLM_VECTOR_ROUND = 0 # 0 to disable rounding
EMBED_MODEL = SentenceTransformer(LLM_MODEL)
KEYBERT_MODEL = KeyBERT(model=EMBED_MODEL)

# Console and constants
console = Console(log_time=True, log_path=False)
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
max_workers = round(os.cpu_count() * 0.9)

# Limit one thread-pool per process (we keep only one process now)
torch.set_num_threads(max_workers)               # PyTorch kernels
os.environ["OMP_NUM_THREADS"]  = str(max_workers)  # OpenMP
os.environ["MKL_NUM_THREADS"]  = str(max_workers)  # Intel MKL (fallback)

visited = set()
queue = []
total_pages = 0

# --- Utilities ---

def sanitize_filename(title: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', ' ', title)[:255]

def build_page_url(title: str) -> str:
    import urllib.parse
    return f"{WIKI_URL.replace('/api.php', '')}/index.php?title=" + urllib.parse.quote(title.replace(" ", "_"))

def html_to_plaintext(html: str) -> str:
    """Remove tags/scripts and return readable plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)

def html_to_markdown(html: str) -> str:
    """
    Convert full HTML to Markdown.
    Uses ATX headings so <h2> ➜ ##, <h3> ➜ ###, etc.
    """
    return md(html, heading_style="ATX", strip=['style', 'script']).strip()

def _get_item_title(obj: dict | str) -> str:
    """
    MediaWiki v1 ⇒   {'*': 'Foo'}
    MediaWiki v2 ⇒   {'title': 'Foo'}  or {'name': 'Foo'}
    Accept either form. Falls back to str(obj) for robustness.
    """
    if isinstance(obj, dict):
        return obj.get("*") or obj.get("title") or obj.get("name") or ""
    return str(obj)

class EMATimeRemainingColumn(TimeRemainingColumn):
    """ETA column with exponential moving-average smoothing."""
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self._ema_rate: Optional[float] = None  # items/sec

    def render(self, task):
        if task.completed == 0 or task.elapsed is None:
            return super().render(task)

        inst_rate = task.completed / task.elapsed if task.elapsed else 0
        self._ema_rate = (
            inst_rate if self._ema_rate is None
            else self.alpha * inst_rate + (1 - self.alpha) * self._ema_rate
        )

        if not self._ema_rate:
            return super().render(task)

        remaining = (task.total - task.completed) / self._ema_rate
        style_kw = {"style": self.style} if hasattr(self, "style") else {}
        return Text(format_time(remaining), **style_kw)

# --- Parser ---

def parse_html_into_summary_and_sections(html: str) -> dict:
    """
    Split rendered HTML into:
        summary      : text before the first <h2> or <h3>
        sections     : top-level <h2> blocks
        subsections  : nested <h3> blocks inside each section

    Returned schema:
    {
        "summary": str,
        "sections": [
            {
                "heading": str,
                "content": str,
                "subsections": [
                    {"heading": str, "content": str},
                    ...
                ]
            },
            ...
        ]
    }
    """
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("div", class_="mw-parser-output") or soup

    result = {"summary": "", "sections": []}
    current_section: dict | None = None
    current_subsection: dict | None = None

    for node in root.children:
        # skip pure whitespace strings
        if isinstance(node, str) and not node.strip():
            continue

        tag = getattr(node, "name", None)

        # ── Headings ───────────────────────────────────────────────────────────
        if tag in ("h2", "h3"):
            heading_text = node.get_text(separator=" ", strip=True)
            level = 2 if tag == "h2" else 3

            if level == 2:                          # new section
                # flush an open subsection, if any
                if current_subsection and current_section:
                    current_subsection["content"] = current_subsection["content"].strip()
                    current_section["subsections"].append(current_subsection)
                    current_subsection = None

                # flush the previous section
                if current_section:
                    current_section["content"] = current_section["content"].strip()
                    result["sections"].append(current_section)

                # start a clean section
                current_section = {
                    "heading": heading_text,
                    "content": "",
                    "subsections": []
                }

            else: # new subsection (h3)
                # if no section yet, create a placeholder
                if current_section is None:
                    current_section = {
                        "heading": "Unnamed Section",
                        "content": "",
                        "subsections": []
                    }

                # flush previous subsection
                if current_subsection:
                    current_subsection["content"] = current_subsection["content"].strip()
                    current_section["subsections"].append(current_subsection)

                # start subsection
                current_subsection = {
                    "heading": heading_text,
                    "content": ""
                }

            continue  # heading handled, move to next node

        # ── Non-heading content ───────────────────────────────────────────────
        text_block = node.get_text(separator=" ", strip=True) if hasattr(node, "get_text") else str(node).strip()
        if not text_block:
            continue

        if current_subsection is not None:
            current_subsection["content"] += text_block + "\n"
        elif current_section is not None:
            current_section["content"] += text_block + "\n"
        else:
            result["summary"] += text_block + "\n"

    # ── Final flushes ─────────────────────────────────────────────────────────
    if current_subsection:
        current_subsection["content"] = current_subsection["content"].strip()
        if current_section is None:
            current_section = {
                "heading": "Unnamed Section",
                "content": "",
                "subsections": []
            }
        current_section["subsections"].append(current_subsection)

    if current_section:
        current_section["content"] = current_section["content"].strip()
        result["sections"].append(current_section)

    result["summary"] = result["summary"].strip()

    # trim subsection whitespace
    for sec in result["sections"]:
        for sub in sec["subsections"]:
            sub["content"] = sub["content"].strip()

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
    """
    Retrieve a single wiki page as rendered HTML, convert it to Markdown,
    extract plain-text for embeddings, and return the standard project schema.

    Returns a dict with:
        title, url, markdown, summary, sections, categories,
        links, raw_text, and the usual blacklist / error flags.
    """
    # Title blacklist
    if any(prefix.lower() in title.lower() for prefix in PAGE_BLACKLIST):
        return {"title": title, "blacklisted": True}

    params = {
        "action": "parse",
        "page": title,
        "prop": "text|categories|links",
        "format": "json",
        "formatversion": 2,
    }
    if API_KEY:
        params["apikey"] = API_KEY

    try:
        data = requests.get(WIKI_URL, params=params, timeout=30).json()
    except Exception:
        return {"title": title, "fetch_error": True}

    if "error" in data or "parse" not in data:
        return {"title": title, "fetch_error": True}

    parse = data["parse"]
    raw_html: str = parse.get("text", "")
    categories = [_get_item_title(c) for c in parse.get("categories", [])]
    links_raw = [_get_item_title(l) for l in parse.get("links", [])]

    # Category blacklist
    if any(bc.lower() in c.lower() for bc in CATEGORY_BLACKLIST for c in categories):
        return {"title": title, "blacklisted": True}

    # Convert formats
    markdown = html_to_markdown(raw_html)
    plain_text = html_to_plaintext(raw_html)
    structure = parse_html_into_summary_and_sections(raw_html)

    # Filter internal links
    links = [
        lt for lt in links_raw
        if not any(p.lower() in lt.lower() for p in PAGE_BLACKLIST)
    ]

    return {
        "title":      title,
        "url":        build_page_url(title),
        "markdown":   markdown,
        "summary":    structure["summary"],
        "sections":   structure["sections"],
        "categories": categories,
        "links":      links,
        "raw_text":   plain_text,
    }

# --- Phase 1: Crawl ---

def crawl_wiki(pages: list = None, verbose: bool = False) -> dict:
    if pages is None:
        pages = fetch_all_pages()

    output_pages = []
    category_map = {}
    wiki_name = fetch_wiki_name()

    # --- Progress bar for page fetching -------------------------------------
    with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            EMATimeRemainingColumn(alpha=0.2),     # same steady ETA
            console=console,
            refresh_per_second=3,                 # same refresh rate
    ) as progress:
        task = progress.add_task("Processing pages...", total=len(pages))

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

            # update category map
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
    flat_sections = []
    embeddings = []

    for page in data:
        page_title = page["title"]
        for section in page.get("sections", []):
            if "semantic_embedding" not in section:
                continue                    # skip short sections

            section_title = section.get("heading", "Unnamed Section")
            full_label = f"{page_title} - {section_title}"

            flat_sections.append({
                "page_title": page_title,
                "section_title": section_title,
                "title": full_label,
                "semantic_embedding": section["semantic_embedding"],
                "ref": section,
            })
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
    # Scale with length, but cap at 15
    top_n = min(15, max(5, round(len(text) / 100)))

    keywords = KEYBERT_MODEL.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_maxsum=True,
        top_n=top_n
    )

    return [kw for kw, _ in keywords]

def enrich_page(page: dict) -> dict:
    text_length = len(page.get("raw_text", ""))

    # Always embed the summary
    page["semantic_embedding"] = generate_embedding(page["summary"])

    # Generate keywords from the whole raw text
    page["keywords"] = extract_keywords(page["raw_text"])

    # Skip section-level embedding if short
    if text_length < 3000:
        return page

    for section in page.get("sections", []):
        combined_text = f"{section['heading']}\n\n{section['content']}".strip()
        section["semantic_embedding"] = generate_embedding(combined_text)
        section["full_text"] = combined_text

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
        task = progress.add_task("[cyan]Generating cluster labels...", total=len(cluster_texts))

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
                cluster_labels[int(cluster_id)] = ", ".join(keywords)
            except Exception as e:
                cluster_labels[int(cluster_id)] = f"cluster_{int(cluster_id)}"
                console.log(f"[yellow]Warning: Failed to label cluster {int(cluster_id)}: {e}")

            progress.update(task, advance=1)

    return cluster_labels

def clean_data(pages: list[dict], sections: list[dict]):
    for page in pages:
        page.pop("semantic_embedding", None)
        page.pop("raw_text", None)
        for section in page.get("sections", []):
            section.pop("semantic_embedding", None)
            section.pop("full_text", None)

    for section in sections:
        section.pop("semantic_embedding", None)

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
    enriched_pages = []

    with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            EMATimeRemainingColumn(alpha=0.2),  # steadier ETA
            console=console,
            refresh_per_second=3,  # redraw every 0.33 s
    ) as progress:
        task = progress.add_task("Enriching pages...", total=len(raw_data["pages"]))

        for page in raw_data["pages"]:
            result = enrich_page(page)
            if args.verbose:
                progress.log(f"[green]Enriched:[/] {result['title']}", highlight=False)

            enriched_pages.append(result)
            progress.update(task, advance=1)

    # Step 3–7: Clustering, labeling, and similarity
    flat_sections, section_embeddings = extract_all_section_embeddings(enriched_pages)
    section_cluster_ids = cluster_sections(section_embeddings, n_clusters=20)
    assign_clusters_to_sections(flat_sections, section_cluster_ids)

    cluster_labels = generate_cluster_labels(flat_sections, section_cluster_ids)
    for section, cid in zip(flat_sections, section_cluster_ids):
        section["ref"]["cluster_label"] = cluster_labels[cid]

    console.log("[blue]Generating related pages...")
    generate_related_pages(enriched_pages, N=5)
    console.log("[blue]Generating related sections...")
    generate_related_pages(flat_sections, N=5)

    # Step 8: Strip vectors for storage
    clean_data(enriched_pages, flat_sections)

    # Step 9: Output
    wiki_name = args.wiki_name or raw_data["wiki_name"]
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_path = args.output or f"{wiki_name} Data {date_str}.json"

    final_data = {
        "wiki_name": wiki_name,
        "content_format": "markdown",
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
    start_time = time.time()

    try:
        main()
    except KeyboardInterrupt:
        console.log("[red]Process interrupted by user. Exiting.")
        exit(1)

    # Compute duration
    duration = int(time.time() - start_time)
    days, rem = divmod(duration, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)

    # Build output string conditionally (no seconds)
    parts = []
    if days: parts.append(f"{days}d")
    if hours: parts.append(f"{hours}h")
    if minutes: parts.append(f"{minutes}m")
    if not parts: parts.append("0m")  # default if < 60 sec

    console.log(f"[green]Done in: {' '.join(parts)}")