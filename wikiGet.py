# ────────────────────────── Standard library ──────────────────────────
import argparse
import json
import os
import re
import time
from math import sqrt
from datetime import datetime
from typing import Optional

# ───────────────────────── Third-party packages ────────────────────────
import numpy as np
import requests
import torch
from bs4 import BeautifulSoup, Comment
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
max_workers = os.cpu_count()

# Limit one thread-pool per process (we keep only one process now)
torch.set_num_threads(max_workers)               # PyTorch kernels
os.environ["OMP_NUM_THREADS"]  = str(max_workers)  # OpenMP
os.environ["MKL_NUM_THREADS"]  = str(max_workers)  # Intel MKL (fallback)

visited = set()
queue = []
total_pages = 0

# Regexes reused later
EDIT_TAG_RE   = re.compile(r"\[\s?edit\s?]", flags=re.I)
IMAGE_MD_RE   = re.compile(r"!\[[^]]*]\([^)]*\)")   # Markdown images ![alt](url)
REF_NUM_RE    = re.compile(r"\[\d+]")                # [1] reference markers

# --- Utilities ---

def sanitize_filename(title: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', ' ', title)[:255]

def build_page_url(title: str) -> str:
    import urllib.parse
    return f"{WIKI_URL.replace('/api.php', '')}/index.php?title=" + urllib.parse.quote(title.replace(" ", "_"))

def clean_article_html(html: str) -> str:
    """
    Remove interface chrome, comments, *all* <img> elements, and unwrap <a>
    tags so only their label text remains. Returns HTML ready for Markdown
    conversion.
    """
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("div", class_="mw-parser-output") or soup

    # ── Delete interface / maintenance blocks
    selectors = (
        ".mw-editsection,"
        "#toc,"
        "#catlinks,"
        "table.navbox,"
        "table.ambox,"
        "div.hatnote,"
        "span.citation-needed,"
        "sup.Inline-Template"
    )
    for tag in root.select(selectors):
        tag.decompose()

    # ── Remove images completely (no alt text) and orphan <a> that contain only an img
    for img in root.find_all("img"):
        parent = img.parent
        img.decompose()
        if parent.name == "a" and not parent.get_text(strip=True):
            parent.decompose()

    # ── Strip HTML comments
    for comment in root.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # ── Unwrap hyperlinks: keep label, drop href
    for a_tag in root.find_all("a"):
        text = a_tag.get_text(" ", strip=True)
        if text:
            a_tag.replace_with(text)
        else:
            a_tag.decompose()  # link with no visible label

    return str(root)

def html_to_markdown(html: str) -> str:
    """
    Convert cleaned HTML to GitHub-flavoured Markdown and scrub leftover
    image syntax, [edit] links, and numeric reference markers.
    """
    md_text = md(
        html,
        heading_style="ATX",
        strip=["style", "script"],   # drops embedded CSS/JS blocks
    )

    md_text = IMAGE_MD_RE.sub("", md_text)
    md_text = EDIT_TAG_RE.sub("", md_text)
    md_text = REF_NUM_RE.sub("", md_text)

    # Collapse excess blank lines produced by tag removal
    md_text = re.sub(r"\n{3,}", "\n\n", md_text).strip()
    return md_text


def html_to_plaintext(html: str, sep: str = "\n") -> str:
    """
    Return readable plain text from cleaned HTML. Uses the same cleaner so
    plaintext and Markdown stay in sync.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=sep, strip=True)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _get_item_title(obj: dict | str | None) -> str:
    """
    Return a readable string from MediaWiki link/category objects.
    Replaces '_' with ' ' and falls back to "" if everything is missing.
    """
    if obj is None:
        return ""
    if isinstance(obj, dict):
        title = (
            obj.get("*")
            or obj.get("title")
            or obj.get("name")
            or obj.get("category")
            or ""
        )
    else:
        title = str(obj)

    return title.replace("_", " ")

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
    Retrieve one wiki page, clean the HTML, convert to Markdown, and return:

        {
            "title":      <page title>,
            "url":        <full wiki URL>,
            "content":    <clean markdown>,
            "categories": [category names without "Category:"],
            "links":      [page links without "Category:"],
        }

    Skips pages or categories that match the PAGE_BLACKLIST or CATEGORY_BLACKLIST.
    """

    # ------------------------------------------------------------------ guard
    if any(prefix.lower() in title.lower() for prefix in PAGE_BLACKLIST):
        return {"title": title, "blacklisted": True}

    # ----------------------------------------------------------------- fetch
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

    # -------------------------------------------------- collect & cleanse
    # Original names from the API
    raw_categories = [_get_item_title(c) for c in parse.get("categories", [])]
    raw_links      = [_get_item_title(l) for l in parse.get("links", [])]

    # Remove "Category:" prefix where present
    strip_cat = lambda t: t[len("Category:") :] if t.lower().startswith("category:") else t
    categories = [strip_cat(c) for c in raw_categories]
    links      = [strip_cat(l) for l in raw_links]

    # Category blacklist (check against original names)
    if any(bc.lower() in c.lower() for bc in CATEGORY_BLACKLIST for c in raw_categories):
        return {"title": title, "blacklisted": True}

    # Filter out page-namespace prefixes in links
    links = [l for l in links if not any(p.lower() in l.lower() for p in PAGE_BLACKLIST)]

    # -------------------------------------------------- HTML → Markdown
    clean_html = clean_article_html(raw_html)
    markdown   = html_to_markdown(clean_html)

    # -------------------------------------------------- final payload
    return {
        "title":      title,
        "url":        build_page_url(title),
        "content":    markdown,
        "categories": categories,
        "links":      links,
    }

def cluster_pages(
    pages: list[dict],
    n_clusters: int | None = None,
    random_state: int = 42,
) -> dict[int, list[int]]:
    """
    Group pages by similarity of their `semantic_embedding`.

    Parameters
    ----------
    pages        : list of page dicts (each must have "semantic_embedding").
    n_clusters   : optional; if None, picks ⌈√(N/2)⌉ – a common heuristic
                   that gives coarse thematic buckets without over-splitting.
    random_state : passed to KMeans for deterministic results.

    Returns
    -------
    cluster_index : dict {cluster_id: [page_index, …]}  (order = input order)

    Side effects
    ------------
    Adds an int field `page["cluster"]` to every page dict.
    """

    if not pages:
        return {}

    # ── choose cluster count automatically if not supplied
    if n_clusters is None:
        n_clusters = max(2, int(np.ceil(sqrt(len(pages) / 2))))

    # ── gather embeddings
    embeddings = np.array([p["semantic_embedding"] for p in pages])

    # ── run K-Means
    km = KMeans(
        n_clusters=n_clusters,
        n_init="auto",
        random_state=random_state,
    )
    labels = km.fit_predict(embeddings)

    # ── attach cluster IDs & build quick lookup dict
    cluster_index: dict[int, list[int]] = {}
    for idx, (page, cid) in enumerate(zip(pages, labels)):
        cid_int = int(cid)  # avoid NumPy int subclasses
        page["cluster"] = cid_int
        cluster_index.setdefault(cid_int, []).append(idx)

    return cluster_index

# --- Phase 1: Crawl ---

def crawl_wiki(pages: list = None, verbose: bool = False, test: bool = False) -> dict:
    if pages is None:
        pages = fetch_all_pages()

    if test:
        pages = pages[:10]

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
    """
    Attach a semantic embedding and keyword list to a page that now has only
    one text field, `content`.

    Adds:
        page["semantic_embedding"] : list[float]
        page["keywords"]           : list[str]
    """
    text = page.get("content", "")
    if not text:
        return page    # empty stub page; nothing to enrich

    # Whole-page embedding
    page["semantic_embedding"] = generate_embedding(text)

    # Keywords from the whole text (KeyBERT auto-scales top_n internally)
    page["keywords"] = extract_keywords(text)

    return page

def clean_data(pages: list[dict]) -> None:
    """
    Remove large, runtime-only fields before JSON export.
    Currently we keep just the markdown `content`, `keywords`,
    and metadata; drop the dense embedding to shrink the file.
    """
    for page in pages:
        page.pop("semantic_embedding", None)

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Semantic Wiki Processor")
    parser.add_argument("-o", "--output", help="Output .json path")
    parser.add_argument("-w", "--wiki-name", help="Override wiki name")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--workers", type=int, help="CPU threads to use")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Process only the first 10 pages for a quick end-to-end test",
    )
    args = parser.parse_args()

    # Step 1: Crawl wiki content
    if args.test_run:
        console.log("[yellow]Test-run mode: limiting to 10 pages.")

    raw_data = crawl_wiki(verbose=args.verbose, test=args.test_run)

    # ── Decide output path *before* any enrichment ─────────────────────────
    wiki_name  = args.wiki_name or raw_data["wiki_name"]
    date_str   = datetime.now().strftime("%Y-%m-%d")
    output_path = args.output or f"{wiki_name} Data {date_str} Clean.json"

    # ── Write pre-embedding snapshot --------------------------------------
    pre_snapshot = {
        "wiki_name":       wiki_name,
        "content_format":  "markdown",
        "pages":           raw_data["pages"],     # cleaned content only
        "categories":      raw_data["categories"],
        "note":            "PRE-EMBEDDING SNAPSHOT — embeddings, keywords, "
                           "clusters will be added in the final pass."
    }

    # make parent folder if supplied path includes a directory
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pre_snapshot, f, ensure_ascii=False, indent=2)

    console.log(f"[cyan]Pre-embedding snapshot saved to: {output_path}", highlight=False)

    # Step 2: Enrich pages
    enriched_pages = []

    with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            EMATimeRemainingColumn(alpha=0.2),  # steadier ETA
            console=console,
            refresh_per_second=2,  # redraw every 0.33 s
    ) as progress:
        task = progress.add_task("Enriching pages...", total=len(raw_data["pages"]))

        for page in raw_data["pages"]:
            enrich_start_time = time.perf_counter()

            result = enrich_page(page)

            elapsed = time.perf_counter() - enrich_start_time
            if args.verbose:
                progress.log(
                    f"[green]Enriched:[/] {result['title']} "
                    f"([cyan]{elapsed:.1f}s[/])",
                    highlight=False,
                )

            enriched_pages.append(result)
            progress.update(task, advance=1)

    # ── Step 3: Page-level related-pages and clustering ────────────────────
    console.log("[blue]Generating related pages...")
    generate_related_pages(enriched_pages, N=5)  # similarity on page embeddings

    console.log("[blue]Clustering pages...")
    cluster_index = cluster_pages(enriched_pages)  # adds page["cluster"]

    # ── Step 4: Strip vectors before storage ───────────────────────────────
    clean_data(enriched_pages)  # new, page-only version

    # ── Step 5: Output ─────────────────────────────────────────────────────
    final_data = {
        "wiki_name": wiki_name,
        "content_format": "markdown",
        "embedding_model": f"sentence-transformers/{LLM_MODEL}",
        "pages": enriched_pages,
        "categories": raw_data["categories"],
        "clusters": cluster_index,  # {cluster_id: [page_indices]}
    }

    output_path = args.output or f"{wiki_name} Data {date_str}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    console.log(f"[green]Final data saved to: {output_path}", highlight=False)

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

    console.log(f"[green]Done in: {' '.join(parts)}", highlight=False)