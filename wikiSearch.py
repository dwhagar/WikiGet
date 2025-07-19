#!/usr/bin/env python3
"""
wikiSearch.py

A script to perform multi-stage expansion and semantic filtering of wiki pages based on:
1. Title matching
2. Outgoing link expansion
3. Category expansion + outgoing links
4. Cluster expansion (keyword & semantic)
5. Full page-level semantic filtering with progress bar and embedding cache
Produces a JSON export with the merged, deduplicated final pool of pages and rebuilt indexes.
"""
import argparse
import json
import sys
import re
from typing import List, Iterable, Dict, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()        # suppress warnings, keep errors

# ----------------------------------------------------------------------------
# Global configuration & shared utilities
# ----------------------------------------------------------------------------
LLM_MODEL = "all-mpnet-base-v2"
EMBED_MODEL = SentenceTransformer(LLM_MODEL)
console = Console(log_time=True, log_path=False)

# Format seconds into H:MM:SS or MM:SS
def format_time(seconds: float | None) -> str:
    """
    Convert seconds to H:MM:SS (if hours>0) or MM:SS string.
    """
    if seconds is None or seconds == float("inf"):
        return "--:--:--"
    sec = int(seconds + 0.5)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

# Default max tokens for splitting long text
default_max_length = getattr(EMBED_MODEL.tokenizer, "model_max_length", 512)

# ----------------------------------------------------------------------------
# Data loading and mapping
# ----------------------------------------------------------------------------
def load_data(path: str) -> Dict:
    """
    Load JSON data produced by wikiGet.py.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_queries() -> List[str]:
    """
    Prompt for search terms (one per line) until blank, return list.
    """
    console.log("[cyan]Enter search terms, one per line. Blank line to execute.", highlight=False)
    queries: List[str] = []
    for line in sys.stdin:
        term = line.strip()
        if not term:
            break
        queries.append(term)
    return queries


def build_page_maps(pages: List[Dict]) -> (Dict[str, Dict], Dict[str, List[str]]):
    """
    From all pages, build maps: title->page, category->titles.
    """
    pages_by_title: Dict[str, Dict] = {}
    category_map: Dict[str, List[str]] = {}
    for p in pages:
        title = p.get("title", "")
        pages_by_title[title] = p
        for cat in p.get("categories", []):
            category_map.setdefault(cat, []).append(title)
    return pages_by_title, category_map

# ----------------------------------------------------------------------------
# Expansion stages
# ----------------------------------------------------------------------------
def get_title_seeds(term: str, pages_by_title: Dict[str, Dict]) -> List[Dict]:
    """
    Stage 0: Pages where title contains or is contained in term.
    """
    term_lower = term.lower()
    seeds: List[Dict] = []
    for title, page in pages_by_title.items():
        t = title.lower()
        if term_lower in t or t in term_lower:
            seeds.append(page)
    return seeds


def expand_links(seeds: List[Dict], pages_by_title: Dict[str, Dict], pool: Dict[str, Dict]) -> int:
    """
    Stage 1: One-hop outgoing link expansion from seeds.
    """
    added = 0
    for seed in seeds:
        for link_title in seed.get("links", []):
            linked = pages_by_title.get(link_title)
            if linked and linked.get("key") not in pool:
                pool[linked["key"]] = linked
                added += 1
    return added


def expand_categories(seeds: List[Dict], category_map: Dict[str, List[str]],
                      pages_by_title: Dict[str, Dict], pool: Dict[str, Dict]) -> (int, int):
    """
    Stage 2: Add pages in each seed's categories, then one-hop links from those.
    """
    category_pages: List[Dict] = []
    for seed in seeds:
        for cat in seed.get("categories", []):
            for title in category_map.get(cat, []):
                page = pages_by_title.get(title)
                if page and page.get("key") not in pool:
                    pool[page["key"]] = page
                    category_pages.append(page)
    cat_added = len(category_pages)

    link_added = 0
    for page in category_pages:
        for link_title in page.get("links", []):
            linked = pages_by_title.get(link_title)
            if linked and linked.get("key") not in pool:
                pool[linked["key"]] = linked
                link_added += 1
    return cat_added, link_added

# ----------------------------------------------------------------------------
# Cluster-based expansion
# ----------------------------------------------------------------------------
def select_clusters_by_keyword(term: str, clusters: Dict[str, Dict]) -> set:
    """
    Pick clusters whose keyword list contains term.
    """
    term_lower = term.lower()
    matched = set()
    for cid, info in clusters.items():
        for kw_item in info.get("keywords", []):
            if term_lower in kw_item[0].lower():
                matched.add(cid)
                break
    return matched


def select_clusters_by_semantic(term: str, clusters: Dict[str, Dict], threshold: float) -> set:
    """
    Encode term and compare to each cluster centroid, include above threshold.
    """
    query_vec = EMBED_MODEL.encode([term])[0].reshape(1, -1)
    matched = set()
    for cid, info in clusters.items():
        centroid = np.array(info.get("centroid", [])).reshape(1, -1)
        sim = cosine_similarity(query_vec, centroid)[0][0]
        if sim >= threshold:
            matched.add(cid)
    return matched


def expand_clusters(cluster_ids: Iterable[str], clusters: Dict[str, Dict],
                    pages_by_title: Dict[str, Dict], pool: Dict[str, Dict]) -> int:
    """
    Stage 3: Add every page from selected clusters into pool.
    """
    added = 0
    for cid in cluster_ids:
        for page_ref in clusters[cid].get("pages", []):
            title = page_ref.get("title")
            page = pages_by_title.get(title)
            if page and page.get("key") not in pool:
                pool[page["key"]] = page
                added += 1
    return added

# ----------------------------------------------------------------------------
# Long text splitting utility
# ----------------------------------------------------------------------------
def split_long_text(text: str, tokenizer, target_tokens: int) -> List[str]:
    """
    Break text into <= target_tokens chunks, preferring blank-line splits.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current: List[str] = []
    cur_tok = 0
    for para in paragraphs:
        p_tok = len(tokenizer.encode(para, add_special_tokens=False))
        if cur_tok + p_tok > target_tokens and current:
            chunks.append("\n\n".join(current))
            current, cur_tok = [], 0
        if p_tok > target_tokens:
            ids = tokenizer.encode(para, add_special_tokens=False)
            for i in range(0, len(ids), target_tokens):
                chunk = tokenizer.decode(ids[i: i + target_tokens])
                chunks.append(chunk)
            continue
        current.append(para)
        cur_tok += p_tok
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text[: target_tokens * 4]]

# ----------------------------------------------------------------------------
# Progress bar ETA with EMA smoothing
# ----------------------------------------------------------------------------
class EMATimeRemainingColumn(TimeRemainingColumn):
    """ETA column with exponential moving-average smoothing."""
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self._ema_rate: Optional[float] = None
    def render(self, task):
        if task.completed == 0 or task.elapsed is None:
            return super().render(task)
        inst_rate = task.completed / task.elapsed if task.elapsed else 0
        self._ema_rate = inst_rate if self._ema_rate is None else (
            self.alpha * inst_rate + (1 - self.alpha) * self._ema_rate
        )
        if not self._ema_rate:
            return super().render(task)
        remaining = (task.total - task.completed) / self._ema_rate
        return Text(format_time(remaining))

# ----------------------------------------------------------------------------
# Full semantic filtering stage with progress bar and cache
# ----------------------------------------------------------------------------
def semantic_filter_with_progress(term: str,
                                  pool: Dict[str, Dict],
                                  embed_cache: Dict[str, np.ndarray],
                                  threshold: float) -> Dict[str, Dict]:
    """
    Stage 4: Full page-level semantic filtering using caching and rich progress bar.
    """
    survivors: Dict[str, Dict] = {}
    pages_list = list(pool.values())
    total = len(pages_list)
    tok = EMBED_MODEL.tokenizer
    max_len = default_max_length
    target_len = max_len - 8
    query_vec = EMBED_MODEL.encode([term])[0].reshape(1, -1)

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        EMATimeRemainingColumn(alpha=0.2),
        console=console,
        refresh_per_second=3,
    ) as progress:
        task = progress.add_task(f"Embedding pages for '{term}'", total=total)
        for page in pages_list:
            key = page.get("key")
            if key not in embed_cache:
                content = page.get("content", "")
                tokens = tok.encode(content, add_special_tokens=False)
                chunks = split_long_text(content, tok, target_len) if len(tokens) > max_len else [content]
                vectors = EMBED_MODEL.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
                emb = vectors.mean(axis=0) if len(vectors) > 1 else vectors[0]
                embed_cache[key] = emb
            else:
                emb = embed_cache[key]
            sim = cosine_similarity(query_vec, emb.reshape(1, -1))[0][0]
            if sim >= threshold:
                survivors[key] = page
            progress.advance(task)
    return survivors

# ----------------------------------------------------------------------------
# Output building
# ----------------------------------------------------------------------------
def build_output_indexes(pages: List[Dict]) -> (Dict[str, List[str]], Dict[str, List[str]]):
    """
    Build final categories and keyword_index maps.
    """
    categories: Dict[str, List[str]] = {}
    keyword_index: Dict[str, List[str]] = {}
    for page in pages:
        title = page.get("title", "")
        for cat in page.get("categories", []):
            categories.setdefault(cat, []).append(title)
        for kw in page.get("keywords", []):
            keyword_index.setdefault(kw, []).append(title)
    return categories, keyword_index

def write_output(output: Dict, path: Optional[str] = None) -> None:
    """
    Write JSON output to file or stdout.
    """
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        console.log(f"[green]Output written to: {path}", highlight=False)
    else:
        json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
        console.log("", highlight=False)

# ----------------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="WikiSearch with full semantic filtering and merged pools"
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to JSON data file from wikiGet.py")
    parser.add_argument("-o", "--output",
                        help="Path for merged output (defaults to stdout)")
    parser.add_argument("--semantic-threshold", type=float, default=0.25,
                        help="Cosine similarity cutoff for page filtering")
    args = parser.parse_args()

    data = load_data(args.input)
    all_pages = data.get("pages", [])
    clusters = data.get("clusters", {})
    pages_by_title, category_map = build_page_maps(all_pages)

    embed_cache: Dict[str, np.ndarray] = {}

    console.log(f"[cyan]Total pages in dataset: {len(all_pages)}", highlight=False)
    queries = read_queries()

    combined_pool: Dict[str, Dict] = {}
    duplicates_removed = 0

    for term in queries:
        console.log(f"[magenta]\nSearching for '{term}'", highlight=False)
        pool: Dict[str, Dict] = {}

        seeds = get_title_seeds(term, pages_by_title)
        for p in seeds:
            pool[p.get("key")] = p
        console.log(f"[cyan]Stage 0 (title match): seeds={len(seeds)}, pool={len(pool)}", highlight=False)

        link_added = expand_links(seeds, pages_by_title, pool)
        console.log(f"[cyan]Stage 1 (link expansion): added={link_added}, pool={len(pool)}", highlight=False)

        cat_added, cat_link_added = expand_categories(seeds, category_map, pages_by_title, pool)
        console.log(f"[cyan]Stage 2a (category expansion): added={cat_added}", highlight=False)
        console.log(f"[cyan]Stage 2b (links from categories): added={cat_link_added}, pool={len(pool)}", highlight=False)

        kw_clusters = select_clusters_by_keyword(term, clusters)
        sem_clusters = select_clusters_by_semantic(term, clusters, args.semantic_threshold)
        all_cluster_ids = kw_clusters.union(sem_clusters)
        cluster_added = expand_clusters(all_cluster_ids, clusters, pages_by_title, pool)
        console.log(f"[cyan]Stage 3 (cluster expansion): added={cluster_added}, pool={len(pool)}", highlight=False)

        before_sem = len(pool)
        survivors = semantic_filter_with_progress(term, pool, embed_cache, args.semantic_threshold)
        kept = len(survivors)
        removed = before_sem - kept
        console.log(f"[green]Stage 4 (semantic): kept={kept}, removed={removed}, pool={kept}", highlight=False)

        for p in survivors.values():
            key = p.get("key")
            if key not in combined_pool:
                combined_pool[key] = p
            else:
                duplicates_removed += 1

    console.log(f"[cyan]\nFinal combined pool: {len(combined_pool)} pages, duplicates removed: {duplicates_removed}", highlight=False)

    final_pages = list(combined_pool.values())
    categories, keyword_index = build_output_indexes(final_pages)
    output = {
        "wiki_name": data.get("wiki_name"),
        "content_format": data.get("content_format"),
        "pages": final_pages,
        "categories": categories,
        "keyword_index": keyword_index,
    }
    write_output(output, args.output)

if __name__ == "__main__":
    main()
