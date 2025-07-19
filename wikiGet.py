# ────────────────────────── Standard library ──────────────────────────
import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Optional
import warnings
import urllib.parse
import concurrent.futures
import hashlib
import random
from collections import Counter

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
from transformers import AutoTokenizer
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()        # suppress warnings, keep errors
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
PAGE_BLACKLIST = ["User:", "Talk:", "Template:", "Special:", "Main Page"]

LLM_MODEL = "all-mpnet-base-v2"
KEYBERT_MODEL = "all-minilm-l6-v2"
LLM_VECTOR_ROUND = 0 # 0 to disable rounding
EMBED_MODEL = SentenceTransformer(LLM_MODEL)

# Console and constants
console = Console(log_time=True, log_path=False)
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
max_workers = os.cpu_count()

# Limit one thread-pool per process (we keep only one process now)
torch.set_num_threads(max_workers)               # PyTorch kernels
torch.set_num_interop_threads(max_workers)
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

def split_long_text(text: str, tokenizer, target_tokens: int) -> list[str]:
    """
    Break `text` into ≤ target_tokens segments, preferring blank-line splits.
    Falls back to hard token slicing when paragraphs exceed the limit.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks, current, cur_tok = [], [], 0

    for para in paragraphs:
        p_tok = len(tokenizer.encode(para, add_special_tokens=False))

        if cur_tok + p_tok > target_tokens and current:
            chunks.append("\n\n".join(current))
            current, cur_tok = [], 0

        if p_tok > target_tokens:                    # paragraph itself too big
            ids = tokenizer.encode(para, add_special_tokens=False)
            for i in range(0, len(ids), target_tokens):
                chunk = tokenizer.decode(ids[i : i + target_tokens])
                chunks.append(chunk)
            continue

        current.append(para)
        cur_tok += p_tok

    if current:
        chunks.append("\n\n".join(current))

    return chunks or [text[: target_tokens * 4]]  # crude fallback

def suggest_batch_size(
    pages: list[dict],
    embed_model,
    ram_gb: int = 16,
    percentile: float = 0.95,
    safety_factor: float = 0.75,   # keep 25 % RAM free
    min_size: int = 4,
    max_size: int = 64,
) -> int:
    """
    • Adds `page["n_tokens"]` to every page (counted once, cached forever).
    • Returns a batch size that fits comfortably in `ram_gb` of system RAM.

    Heuristic: use the `percentile` token length as the typical sequence
    length, assume activations ≈ 2 × (hidden_dim × tokens × 4 bytes),
    and reserve (1 – safety_factor) of RAM for Python/OS overhead.
    """
    if not pages:
        return min_size

    warnings.filterwarnings(
        "ignore",
        message=r"Token indices sequence length is longer than the specified maximum",
        module=r"transformers.tokenization_utils_base",
    )

    # ── resolve tokenizer (reuse if already present) ─────────────────────
    tok = getattr(embed_model, "tokenizer", None)
    if tok is None:
        model_name = (
            getattr(embed_model, "model_name_or_path", None)
            or getattr(embed_model, "model_name", None)
            or "sentence-transformers/"+embed_model.__class__.__name__
        )
        tok = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=True,      # stay offline
        )

    # ── count tokens once and cache in each page ─────────────────────────
    lengths = []
    for p in pages:
        if "n_tokens" not in p:                     # skip if already set
            p["n_tokens"] = len(tok.encode(
                p["content"], add_special_tokens=False
            ))
        lengths.append(p["n_tokens"])

    # ── pick representative sequence length (e.g., 95th percentile) ─────
    seq_len = sorted(lengths)[int(len(lengths) * percentile)]
    hidden_dim = embed_model.get_sentence_embedding_dimension()

    # ── memory budget estimation ─────────────────────────────────────────
    bytes_per_example = hidden_dim * seq_len * 4 * 2        # rough upper-bound
    avail_bytes = int(ram_gb * safety_factor * (1024 ** 3))

    size = max(min_size, avail_bytes // bytes_per_example)
    return min(size, max_size)

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
    # guard against unwanted namespaces
    if any(prefix.lower() in title.lower() for prefix in PAGE_BLACKLIST):
        return {"title": title, "blacklisted": True}

    # fetch raw parse output
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

    # strip out any CSS <style>…</style> blocks
    raw_html = re.sub(r'(?is)<style[^>]*>.*?</style>', '', raw_html)

    # collect & cleanse metadata
    raw_categories = [_get_item_title(c) for c in parse.get("categories", [])]
    raw_links      = [_get_item_title(l) for l in parse.get("links", [])]

    strip_cat = lambda t: t[len("Category:"):] if t.lower().startswith("category:") else t
    categories = [strip_cat(c) for c in raw_categories]
    links      = [strip_cat(l) for l in raw_links]
    if any(bc.lower() in c.lower() for bc in CATEGORY_BLACKLIST for c in raw_categories):
        return {"title": title, "blacklisted": True}
    links = [l for l in links if not any(p.lower() in l.lower() for p in PAGE_BLACKLIST)]

    # HTML → Markdown
    clean_html = clean_article_html(raw_html)
    markdown   = html_to_markdown(clean_html)

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
) -> tuple[dict[int, list[int]], list[list[float]]]:
    """
    Group pages by similarity of their `semantic_embedding`.

    Returns:
      - cluster_index: dict {cluster_id: [page_index, …]}
      - centroids: list of centroid vectors (centroids[i] is for cluster i)
    """
    if not pages:
        return {}, []

    # ── choose cluster count automatically if not supplied
    N = len(pages)
    if n_clusters is None:
        n_clusters = max(2, int(np.ceil(np.sqrt(N / 2))))

    # ── gather embeddings
    embedding_matrix = np.array([p["semantic_embedding"] for p in pages])

    # ── run K-Means
    km = KMeans(
        n_clusters=n_clusters,
        n_init="auto",
        random_state=random_state,
    )
    labels = km.fit_predict(embedding_matrix)

    # ── capture centroids for O(1) lookup by cluster ID
    centroids = km.cluster_centers_.tolist()

    # ── attach labels & build index
    cluster_index: dict[int, list[int]] = {}
    for idx, (page, cid) in enumerate(zip(pages, labels)):
        cid_int = int(cid)
        page["cluster"] = cid_int
        cluster_index.setdefault(cid_int, []).append(idx)

    return cluster_index, centroids

# --- Phase 1: Crawl ---

def threaded_fetch_page_data(title, verbose, progress, task_id):
    data = fetch_page_data(title)

    if verbose:
        if data.get("blacklisted"):
            progress.log(f"[yellow]Blacklisted:[/] {title}", highlight=False)
        elif data.get("fetch_error"):
            progress.log(f"[red]Fetch error:[/] {title}", highlight=False)
        else:
            progress.log(f"[green]Fetched:[/] {title}", highlight=False)

    progress.update(task_id, advance=1)
    return title, data

def crawl_wiki(pages: list = None, verbose: bool = False, test: bool = False) -> dict:
    if pages is None:
        pages = fetch_all_pages()

    if test:
        pages = random.sample(pages, min(10, len(pages)))

    output_pages = []
    category_map = {}
    visited.clear()
    wiki_name = fetch_wiki_name()
    thread_count = max(1, max_workers // 4)

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        EMATimeRemainingColumn(alpha=0.2),
        console=console,
        refresh_per_second=3,
    ) as progress:
        task = progress.add_task("Processing pages...", total=len(pages))

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(threaded_fetch_page_data, title, verbose, progress, task)
                for title in pages
            ]

            for future in concurrent.futures.as_completed(futures):
                title, data = future.result()

                if data.get("blacklisted") or data.get("fetch_error"):
                    visited.add(title)
                    continue

                output_pages.append(data)
                visited.add(title)

                for cat in data.get("categories", []):
                    category_map.setdefault(cat, []).append(title)

    return {
        "pages": output_pages,
        "categories": category_map,
        "wiki_name": wiki_name,
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

def build_clusters_data(pages: list[dict], cluster_index: dict[int, list[int]], centroids) -> dict:
    """
    Construct reverse cluster index with centroid and list of pages (title and key).
    """
    clusters: dict[int, dict] = {}
    for cid, indices in cluster_index.items():
        # ensure centroid is a plain list
        centroid_vec = centroids[cid]
        centroid_list = centroid_vec.tolist() if hasattr(centroid_vec, "tolist") else list(centroid_vec)
        clusters[cid] = {
            "centroid": centroid_list,
            "pages": [
                {"title": pages[i]["title"], "key": pages[i]["key"]}
                for i in indices
            ],
        }
    return clusters

def assign_clusters_and_related(pages: list[dict]) -> None:
    """
    Compute MD5 keys, cluster via cluster_pages, strip embeddings,
    and add a related_pages list of {title, key} for each page.
    """
    # 1) Unique key per page
    for p in pages:
        p["key"] = hashlib.md5(p["content"].encode("utf-8")).hexdigest()

    # 2) Cluster in-place (cluster_pages sets page["cluster"])
    cluster_index, _ = cluster_pages(pages)

    # 3) Promote label to cluster_id
    for p in pages:
        p["cluster_id"] = p.pop("cluster")

    # 4) Remove embeddings now that clustering is done
    for p in pages:
        p.pop("semantic_embedding", None)

    # 5) Build related_pages by looking up peers in the same cluster
    for p in pages:
        peers = cluster_index[p["cluster_id"]]
        p["related_pages"] = [
            {"title": pages[i]["title"], "key": pages[i]["key"]}
            for i in peers
            if pages[i]["key"] != p["key"]
        ]

def summarize_cluster_keywords(cluster_map, page_map, max_keywords=15):
    """
    Returns top keywords per cluster, excluding any that appear only once.

    Args:
        cluster_map (dict[str, list[str]]): cluster_id → list of page titles
        page_map (dict[str, dict]): title → page data with "keywords" field
        max_keywords (int): max number of keywords to return per cluster

    Returns:
        dict[str, list[tuple[str, int]]]: cluster_id → list of (keyword, count)
    """
    cluster_keywords = {}

    for cluster_id, titles in cluster_map.items():
        counter = Counter()

        for title in titles:
            keywords = page_map.get(title, {}).get("keywords", [])
            counter.update(keywords)

        # Filter out keywords with count < 2
        filtered = [(kw, count) for kw, count in counter.items() if count >= 2]

        # Sort and limit to max_keywords
        filtered.sort(key=lambda x: (-x[1], x[0]))
        cluster_keywords[cluster_id] = filtered[:max_keywords]

    return cluster_keywords

def generate_embedding(text: str, decimals=0) -> list:
    raw_vector = EMBED_MODEL.encode(text)
    if decimals < 1:
        return list(raw_vector)
    else:
        return [round(float(v), decimals) for v in raw_vector]

def extract_keywords_parallel(
    pages: list[dict],
    max_workers: int,
    verbose: bool = False,
) -> None:
    """
    Parallel KeyBERT extraction using one-quarter of CPU cores.
    Mutates pages by adding 'keywords' as a list of strings; shows per-page timing logs.
    """
    threads = max(1, max_workers // 4)

    # load once on CPU
    st_model = SentenceTransformer(KEYBERT_MODEL, device="cpu")
    kw_model = KeyBERT(model=st_model)

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        EMATimeRemainingColumn(alpha=0.2),
        console=console,
        refresh_per_second=3,
    ) as progress:
        task = progress.add_task("Extracting keywords...", total=len(pages))

        def process_page(p):
            t0 = time.perf_counter()
            # this returns List[Tuple[str, float]]
            raw = kw_model.extract_keywords(
                p["content"],
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=10,
            )
            elapsed = time.perf_counter() - t0
            # unpack only the keyword strings
            kws = [kw for kw, _ in raw]
            return p, kws, elapsed

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(process_page, p) for p in pages]
            for future in concurrent.futures.as_completed(futures):
                page, kws, elapsed = future.result()
                page["keywords"] = kws
                if verbose:
                    progress.log(
                        f"[green]Keywords extracted:[/] {page['title']} ([cyan]{elapsed:.2f}s[/])",
                        highlight=False,
                    )
                progress.advance(task)

def enrich_pages(
    pages: list[dict],
    batch_size: int = 32,
    decimals: int = 0,
    verbose: bool = False,
) -> None:
    """
    Phase 1: Embedding with progress bar and verbose timing.
    Mutates pages by adding 'semantic_embedding'.
    """
    if not pages:
        return

    tok = EMBED_MODEL.tokenizer
    max_len = getattr(EMBED_MODEL, "max_seq_length", 512)
    target_len = max_len - 8
    items: list[tuple[dict, list[str]]] = []
    for p in pages:
        tokens = tok.encode(p["content"], add_special_tokens=False)
        chunks = (
            split_long_text(p["content"], tok, target_len)
            if len(tokens) > max_len
            else [p["content"]]
        )
        items.append((p, chunks))

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        EMATimeRemainingColumn(alpha=0.2),
        console=console,
        refresh_per_second=3,
    ) as progress:
        task = progress.add_task("Enriching pages...", total=len(items))
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            texts: list[str] = []
            parts: list[int] = []
            for page, chunks in batch:
                parts.append(len(chunks))
                texts.extend(chunks)

            t0 = time.perf_counter()
            vectors = EMBED_MODEL.encode(
                texts,
                batch_size=len(texts),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            batch_elapsed = time.perf_counter() - t0

            total_chunks = sum(parts)
            avg_chunk_time = batch_elapsed / total_chunks if total_chunks else 0.0

            idx = 0
            for (page, _), n_parts in zip(batch, parts):
                vecs = vectors[idx : idx + n_parts]
                idx += n_parts
                emb = vecs[0] if n_parts == 1 else np.mean(vecs, axis=0)
                if decimals < 1:
                    page["semantic_embedding"] = emb.tolist()
                else:
                    page["semantic_embedding"] = [
                        round(float(x), decimals) for x in emb
                    ]

                elapsed = avg_chunk_time * n_parts
                if verbose:
                    info = (
                        f"{n_parts} chunks in {elapsed:.2f}s"
                        if n_parts > 1
                        else f"{elapsed:.2f}s"
                    )
                    progress.log(
                        f"[green]Enriched:[/] {page['title']} ([cyan]{info}[/])",
                        highlight=False,
                    )
                progress.advance(task)

def build_keyword_index(pages: list[dict]) -> dict[str, list[str]]:
    """
    Build an inverted index mapping each keyword to the titles of pages containing it.
    """
    index: dict[str, set[str]] = {}
    for page in pages:
        title = page.get("title", "")
        for kw in page.get("keywords", []):
            index.setdefault(kw, set()).add(title)

    # Convert sets to sorted lists for consistency
    return {kw: sorted(titles) for kw, titles in index.items()}

def clean_data(pages: list[dict]) -> None:
    """
    Remove large, runtime-only fields before JSON export.
    Currently we keep just the markdown `content`, `keywords`,
    and metadata; drop the dense embedding to shrink the file.
    """
    for page in pages:
        page.pop("semantic_embedding", None)
        page.pop("n_tokens", None)

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Semantic Wiki Processor")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("-w", "--wiki-name", help="Override wiki name")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--workers", type=int, help="CPU threads to use")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Process only the first 10 pages for a quick end-to-end test",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Fetch pages only; skip embedding, clustering, and keyword extraction",
    )
    args = parser.parse_args()

    # Phase 0: Fetch pages
    raw_data = crawl_wiki(verbose=args.verbose, test=args.test_run)
    pages = raw_data["pages"]
    categories = raw_data["categories"]

    # Wiki name override
    if args.test_run and not args.wiki_name:
        wiki_name = f"TEST {raw_data["wiki_name"]}"
    elif args.wiki_name:
        wiki_name = args.wiki_name
    else:
        wiki_name = raw_data["wiki_name"]

    # Generate unique keys
    for p in pages:
        p["key"] = hashlib.md5(p["content"].encode("utf-8")).hexdigest()

    # Save clean snapshot
    date_str = datetime.now().strftime("%Y-%m-%d")
    clean_output = args.output or f"{wiki_name} Data {date_str} Clean.json"
    if os.path.dirname(clean_output):
        os.makedirs(os.path.dirname(clean_output), exist_ok=True)
    with open(clean_output, "w", encoding="utf-8") as f:
        json.dump({
            "wiki_name": wiki_name,
            "content_format": "markdown",
            "pages": pages,
            "categories": categories,
        }, f, ensure_ascii=False, indent=2)
    console.log(f"[cyan]Clean data saved to: {clean_output}", highlight=False)

    # If skipping embedding, emit and exit
    if args.no_embedding:
        return

    # Phase 1: Embedding
    batch_size = suggest_batch_size(pages, EMBED_MODEL)
    if args.verbose:
        console.log(f"[cyan]Using batch size:[/] {batch_size}", highlight=False)
    enrich_pages(pages, batch_size=batch_size, verbose=args.verbose)

    # Phase 2: Clustering
    cluster_index, centroids = cluster_pages(pages)
    for p in pages:
        p["cluster_id"] = p.pop("cluster")
    for p in pages:
        p.pop("semantic_embedding", None)
    clusters = build_clusters_data(pages, cluster_index, centroids)

    # Phase 3: Keyword extraction
    worker_count = args.workers or max_workers
    extract_keywords_parallel(pages, max_workers=worker_count, verbose=args.verbose)
    keyword_index = build_keyword_index(pages)

    # Build cluster keywords list
    page_map = {p["title"]: p for p in pages}
    cluster_map = {
        cid: [page["title"] for page in cluster["pages"]]
        for cid, cluster in clusters.items()
    }
    cluster_keywords = summarize_cluster_keywords(cluster_map, page_map)
    for cid, kws in cluster_keywords.items():
        clusters[cid]["keywords"] = kws

    # Final output
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_path = args.output or f"{wiki_name} Data {date_str}.json"
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_data = {
        "wiki_name": wiki_name,
        "content_format": "markdown",
        "pages": pages,
        "categories": categories,
        "clusters": clusters,
        "keyword_index": keyword_index,
    }
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