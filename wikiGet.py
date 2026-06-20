# ────────────────────────── Standard library ──────────────────────────
import argparse
import json
import os
import re
import time
import urllib.parse
import concurrent.futures
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# ───────────────────────── Third-party packages ────────────────────────
import requests
from bs4 import BeautifulSoup, Comment
from markdownify import markdownify as md
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

# Initialize Console early for potential error messages
console = Console(log_time=True, log_path=False)

# ───────────────────────── NLP Dependencies ────────────────────────────
# This script requires spaCy and NLTK for advanced keyword extraction.
# The setup.py handles installation, but you still need to download the spaCy model:
# python -m spacy download en_core_web_sm
try:
    import spacy
    from nltk.corpus import stopwords
except ImportError:
    console.log("[bold red]NLP libraries not found! Please run 'pip install .' in the project directory.[/bold red]")
    exit(1)

# --- NLP Globals ---
STOP_WORDS = None

def initialize_nlp():
    """Load NLP models and stop words into memory."""
    global STOP_WORDS
    console.log("[cyan]Loading NLP model (one-time operation)...[/cyan]", highlight=False)
    try:
        nlp = spacy.load("en_core_web_sm")
        STOP_WORDS = set(stopwords.words("english"))
        console.log("[green]NLP model loaded successfully.[/green]", highlight=False)
        return nlp
    except OSError:
        console.log("[bold red]spaCy model 'en_core_web_sm' not found.[/bold red]")
        console.log("Please run: [bold]python -m spacy download en_core_web_sm[/bold]")
        exit(1)


# ──────────────────────── Rich < 10 fallback ───────────────────────────
try:
    from rich.progress import format_time
except ImportError:
    def format_time(seconds: float | None) -> str:
        if seconds is None or seconds == float("inf"):
            return "--:--:--"
        seconds = int(seconds + 0.5)
        h, rmdr = divmod(seconds, 3600)
        m, s = divmod(rmdr, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

# Regex constants
EDIT_TAG_RE = re.compile(r"\[\s?edit\s?]", flags=re.I)
IMAGE_MD_RE = re.compile(r"!\[[^]]*]\([^)]*\)")
REF_NUM_RE = re.compile(r"\[\d+]")
TABLE_RE = re.compile(r"\|.*\|", re.MULTILINE)


class EMATimeRemainingColumn(TimeRemainingColumn):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self._ema_rate: Optional[float] = None

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


# --- Configuration Loading ---

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a JSON file."""
    default_config = {
        "wiki_url": "https://wiki.moltenaether.com/w/api.php",
        "api_key": "",
        "category_blacklist": [],
        "page_blacklist": [],
        "output_directory": ".",
        "workers": min(32, (os.cpu_count() or 1) * 4),
        "test_run": False,
        "verbose": False,
        "small_page_threshold": 20000,
        "key_phrase_threshold": 20000
    }
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f).get("wikiGet", {})
            default_config.update(config)
            console.log(f"[green]Loaded configuration from: {config_path}", highlight=False)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        console.log(f"[yellow]Could not load {config_path} ({e}), using default settings.", highlight=False)
    return default_config


# --- Utilities & NLP ---

def sanitize_filename(title: str) -> str:
    """Cleans a string to be a valid filename."""
    clean = re.sub(r'[<>:"/\\|?*]', ' ', title)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean[:200]


def create_anchor(title: str) -> str:
    """Creates a URL-safe anchor from a page title."""
    return urllib.parse.quote(re.sub(r'[^a-zA-Z0-9\s-]', '', title).replace(' ', '-').lower())


def extract_keywords_spacy(page: Dict[str, Any], nlp, key_phrase_threshold: int) -> List[str]:
    """Extracts keywords and key phrases using spaCy for high-quality results."""
    content = page['content']
    doc = nlp(content)
    
    lemmas = [
        token.lemma_.lower() for token in doc 
        if token.pos_ in ('NOUN', 'PROPN') and token.lemma_.lower() not in STOP_WORDS and len(token.lemma_) > 2
    ]
    
    num_keywords = 7 + (len(content) // 2500)
    keyword_freq = Counter(lemmas)
    keywords = {word for word, _ in keyword_freq.most_common(num_keywords)}
    
    entities = {ent.text.strip().lower() for ent in doc.ents if len(ent.text.strip()) > 3}
    
    key_phrases = set()
    if len(content) > key_phrase_threshold:
        num_phrases = 5 + (len(content) // 10000)
        long_chunks = [chunk.text.lower() for chunk in doc.noun_chunks if 1 < len(chunk.text.split()) < 6]
        phrase_freq = Counter(long_chunks)
        for phrase, _ in phrase_freq.most_common(num_phrases):
            key_phrases.add(phrase)

    title_doc = nlp(page['title'])
    title_keywords = {
        token.lemma_.lower() for token in title_doc 
        if token.pos_ in ('NOUN', 'PROPN') and token.lemma_.lower() not in STOP_WORDS
    }
    
    all_terms = sorted(list(keywords | key_phrases | entities | title_keywords))
    page['keywords'] = all_terms
    return all_terms


def build_page_url(title: str, wiki_url: str) -> str:
    base = wiki_url.replace('/api.php', '')
    return f"{base}/index.php?title=" + urllib.parse.quote(title.replace(" ", "_"))


def clean_article_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("div", class_="mw-parser-output") or soup
    selectors = (
        ".mw-editsection", "#toc", "#catlinks", "table.navbox",
        "table.ambox", "div.hatnote", "span.citation-needed",
        "sup.Inline-Template"
    )
    for tag in root.select(",".join(selectors)):
        tag.decompose()
    for img in root.find_all("img"):
        parent = img.parent
        img.decompose()
        if parent.name == "a" and not parent.get_text(strip=True):
            parent.decompose()
    for comment in root.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for a_tag in root.find_all("a"):
        text = a_tag.get_text(" ", strip=True)
        if text:
            a_tag.replace_with(text)
        else:
            a_tag.decompose()
    return str(root)


def html_to_markdown(html: str) -> str:
    md_text = md(html, heading_style="ATX", strip=["style", "script"])
    md_text = IMAGE_MD_RE.sub("", md_text)
    md_text = EDIT_TAG_RE.sub("", md_text)
    md_text = REF_NUM_RE.sub("", md_text)
    md_text = re.sub(r"\n{3,}", "\n\n", md_text).strip()
    return md_text


def fetch_all_pages(wiki_url: str, api_key: str) -> List[str]:
    pages = []
    params = {
        "action": "query", "list": "allpages", "format": "json",
        "aplimit": "max", "apfilterredir": "nonredirects"
    }
    if api_key: params["apikey"] = api_key
    apcontinue = None
    while True:
        if apcontinue:
            params["apcontinue"] = apcontinue
        else:
            params.pop("apcontinue", None)
        try:
            response = requests.get(wiki_url, params=params, timeout=30).json()
        except requests.RequestException:
            break
        pages.extend(p["title"] for p in response["query"]["allpages"])
        apcontinue = response.get("continue", {}).get("apcontinue")
        if not apcontinue: break
    return pages


# --- Core Logic ---

def fetch_page_content(title: str, settings: Dict[str, Any]) -> tuple[str, Optional[dict], Optional[str]]:
    if settings['verbose']:
        console.log(f"[dim]Fetching: {title}[/dim]", highlight=False)

    if any(prefix in title for prefix in settings['page_blacklist']):
        return "blacklisted", None, None

    params = {
        "action": "parse", "page": title, "prop": "text|categories",
        "format": "json", "formatversion": 2,
    }
    if settings['api_key']: params["apikey"] = settings['api_key']

    max_retries = 5
    base_delay = 1.0

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(settings['wiki_url'], params=params, timeout=30)
            if resp.status_code in [429, 500, 502, 503, 504] and attempt < max_retries:
                sleep_time = base_delay * (2 ** attempt)
                if settings['verbose']:
                    console.log(f"[yellow]Server error ({resp.status_code}) on '{title}'. Retrying in {sleep_time}s...[/]")
                time.sleep(sleep_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as e:
            if attempt < max_retries:
                sleep_time = base_delay * (2 ** attempt)
                if settings['verbose']:
                    console.log(f"[yellow]Network error on '{title}': {e}. Retrying in {sleep_time}s...[/]")
                time.sleep(sleep_time)
                continue
            else:
                return "error", None, f"Network error after retries: {str(e)}"
    else:
        return "error", None, "Max retries exceeded"

    if "error" in data:
        return "error", None, f"API Error: {data['error'].get('info')}"

    parse = data.get("parse", {})
    if not parse:
        return "error", None, "No parse data found"

    raw_cats = [c.get("category", "") for c in parse.get("categories", [])]
    if any(bc.lower() in c.lower() for bc in settings['category_blacklist'] for c in raw_cats):
        return "blacklisted", None, None

    try:
        raw_html = parse.get("text", "")
        raw_html = re.sub(r'(?is)<style[^>]*>.*?</style>', '', raw_html)
        clean_html = clean_article_html(raw_html)
        markdown = html_to_markdown(clean_html)
    except Exception as e:
        return "error", None, f"Conversion error: {str(e)}"

    return "success", {
        "title": title,
        "content": markdown,
        "categories": raw_cats,
        "url": build_page_url(title, settings['wiki_url'])
    }, None


def save_final_output(pages_data: List[dict], output_dir: Path, date_str: str, settings: Dict[str, Any], nlp, progress: Progress):
    verbose = settings['verbose']
    small_page_threshold = settings['small_page_threshold']
    key_phrase_threshold = settings['key_phrase_threshold']
    total_pages = len(pages_data)

    keyword_task = progress.add_task("[cyan]Generating keywords...", total=total_pages)

    pages_data.sort(key=lambda p: p['title'])
    keyword_map = defaultdict(list)
    for page in pages_data:
        keywords = extract_keywords_spacy(page, nlp, key_phrase_threshold)
        for kw in keywords:
            keyword_map[kw].append(page['title'])
        progress.advance(keyword_task)

    large_pages = [p for p in pages_data if len(p['content']) > small_page_threshold]
    small_pages = [p for p in pages_data if len(p['content']) <= small_page_threshold]

    if verbose:
        console.log(f"  [dim]Large pages: {len(large_pages)}, Small pages: {len(small_pages)}[/dim]", highlight=False)

    # New logic for small pages
    small_page_locations = {}  # title -> filename
    small_pages_by_category = defaultdict(list)

    # Determine the most common category for each small page
    if small_pages:
        all_small_page_cats = [cat for page in small_pages for cat in page.get('categories', [])]
        cat_counts = Counter(all_small_page_cats)

        for page in small_pages:
            page_cats = page.get('categories', [])
            if not page_cats:
                primary_category = "Uncategorized"
            else:
                # Sort page's categories by their global frequency in small pages
                page_cats.sort(key=lambda cat: cat_counts.get(cat, 0), reverse=True)
                primary_category = page_cats[0]
            small_pages_by_category[primary_category].append(page)
    
    for category, pages_in_cat in small_pages_by_category.items():
        cat_filename = f"{sanitize_filename(category)}.md"
        for page in pages_in_cat:
            small_page_locations[page['title']] = cat_filename

    category_map = defaultdict(list)
    for page in pages_data:
        cats = page.get('categories', []) or ["Uncategorized"]
        for cat in cats:
            category_map[cat].append(page['title'])

    def get_page_link(title: str) -> str:
        if title in small_page_locations:
            filename = small_page_locations[title]
            return f"[{filename}#{create_anchor(title)}]"
        else:
            return f"[{sanitize_filename(title)}.md]"

    index_task = progress.add_task("[cyan]Writing index files...", total=3)
    for index_name, data_map, header in [
        ("_Master_Index.md", {p['title']: p for p in pages_data}, "Wiki Master Index"),
        ("_Category_Index.md", category_map, "Wiki Category Index"),
        ("_Keyword_Index.md", keyword_map, "Wiki Keyword Index")
    ]:
        index_path = output_dir / index_name
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(f"# {header}\n**Generated:** {date_str}\n\n---\n\n")
                if index_name == "_Master_Index.md":
                    for title, page in sorted(data_map.items()):
                        link = get_page_link(title).replace('[', f'[{title}](')
                        f.write(f"## {link}\n")
                        f.write(f"- **Categories:** {', '.join(sorted(page.get('categories', []))) or 'N/A'}\n")
                        f.write(f"- **Keywords:** {', '.join(page.get('keywords', [])) or 'N/A'}\n\n")
                else:
                    for key, titles in sorted(data_map.items()):
                        f.write(f"## {key}\n")
                        for title in sorted(titles):
                            link = get_page_link(title).replace('[', f'[{title}](')
                            f.write(f"- {link}\n")
                        f.write("\n")
        except IOError as e:
            progress.console.print(f"[red]Failed to write {index_name}: {e}", highlight=False)
        progress.advance(index_task)

    # New: File-based keyword index
    file_keyword_map = defaultdict(set)
    for page in large_pages:
        filename = f"{sanitize_filename(page['title'])}.md"
        file_keyword_map[filename].update(page['keywords'])
    for category, pages_in_cat in small_pages_by_category.items():
        filename = f"{sanitize_filename(category)}.md"
        for page in pages_in_cat:
            file_keyword_map[filename].update(page['keywords'])

    keyword_index_path = output_dir / "_Keyword_Index.md"
    try:
        with open(keyword_index_path, "a", encoding="utf-8") as f:
            f.write("\n\n---\n\n# Keywords Per File\n\n")
            for filename, keywords in sorted(file_keyword_map.items()):
                f.write(f"## `{filename}`\n")
                if keywords:
                    f.write(f"- {', '.join(sorted(list(keywords)))}\n\n")
                else:
                    f.write("- No keywords found.\n\n")
    except IOError as e:
        progress.console.print(f"[red]Failed to write file-keyword index: {e}", highlight=False)


    large_pages_task = progress.add_task("[cyan]Writing large pages...", total=len(large_pages))
    for page in large_pages:
        page_path = output_dir / f"{sanitize_filename(page['title'])}.md"
        try:
            with open(page_path, "w", encoding="utf-8") as f:
                f.write(f"# {page['title']}\n\n")
                f.write(f"**Original URL:** <{page['url']}>\n")
                f.write(f"**Categories:** {', '.join(sorted(page.get('categories', []))) or 'N/A'}\n")
                f.write(f"**Keywords:** {', '.join(page.get('keywords', [])) or 'N/A'}\n\n---\n\n")
                f.write(page['content'])
        except IOError as e:
            progress.console.print(f"[red]Failed to write page '{page['title']}': {e}", highlight=False)
        progress.advance(large_pages_task)

    # New logic for writing small pages
    if small_pages:
        small_pages_task = progress.add_task("[cyan]Writing small pages...", total=len(small_pages_by_category))
        for category, pages_in_cat in small_pages_by_category.items():
            cat_filename = f"{sanitize_filename(category)}.md"
            cat_path = output_dir / cat_filename
            try:
                with open(cat_path, "w", encoding="utf-8") as f:
                    f.write(f"# Category: {category}\n\n")
                    f.write(f"**Generated:** {date_str}\n\n---\n\n")
                    # Sort pages within the category file
                    pages_in_cat.sort(key=lambda p: p['title'])
                    for page in pages_in_cat:
                        anchor = create_anchor(page['title'])
                        f.write(f"<a name=\"{anchor}\"></a>\n")
                        f.write(f"## Page: {page['title']}\n\n")
                        f.write(f"**Original URL:** <{page['url']}>\n")
                        f.write(f"**Categories:** {', '.join(sorted(page.get('categories', []))) or 'N/A'}\n")
                        f.write(f"**Keywords:** {', '.join(page.get('keywords', [])) or 'N/A'}\n\n---\n\n")
                        f.write(page['content'])
                        f.write(f"\n\n--- END OF PAGE: {page['title']} ---\n\n")
            except IOError as e:
                progress.console.print(f"[red]Failed to write category file '{cat_filename}': {e}", highlight=False)
            progress.advance(small_pages_task)
    
    num_small_files = len(small_pages_by_category)
    console.log(f"[green]Operation complete. Created {len(large_pages) + num_small_files} content files and 3 index files.", highlight=False)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="A tool to download a MediaWiki site into a structured, multi-file format for analysis.")
    parser.add_argument("-c", "--config", default="config.json", help="Path to the configuration file (default: config.json).")
    parser.add_argument("-o", "--output", help="Output directory for the export files. Overrides config file setting.")
    parser.add_argument("--workers", type=int, help="Number of concurrent workers. Overrides config file setting.")
    parser.add_argument("--test-run", action="store_true", help="Process only first 20 pages. Overrides config file setting.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed processing steps. Overrides config file setting.")
    args = parser.parse_args()

    settings = load_config(args.config)
    if args.output: settings['output_directory'] = args.output
    if args.workers: settings['workers'] = args.workers
    if args.test_run: settings['test_run'] = True
    if args.verbose: settings['verbose'] = True

    nlp = initialize_nlp()

    output_dir = Path(settings['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d")

    console.log("[cyan]Fetching page list...", highlight=False)
    pages = fetch_all_pages(settings['wiki_url'], settings['api_key'])
    if settings['test_run']:
        pages = pages[:20]

    collected_data = []
    with Progress(
            TextColumn("[cyan]{task.description}"), BarColumn(), TaskProgressColumn(),
            TimeElapsedColumn(), EMATimeRemainingColumn(alpha=0.2),
            console=console, refresh_per_second=4,
    ) as progress:
        task_id = progress.add_task("Fetching content...", total=len(pages))
        with concurrent.futures.ThreadPoolExecutor(max_workers=settings['workers']) as executor:
            future_to_page = {executor.submit(fetch_page_content, page, settings): page for page in pages}
            for future in concurrent.futures.as_completed(future_to_page):
                page_title = future_to_page[future]
                try:
                    status, data, error = future.result()
                    if status == "success" and data:
                        collected_data.append(data)
                    elif status == "error" and settings['verbose']:
                        progress.console.print(f"[red]Error on {page_title}:[/] {error}", highlight=False)
                except Exception as e:
                    progress.console.print(f"[red]Critical error processing {page_title}:[/] {e}", highlight=False)
                progress.advance(task_id)

        console.log(f"[cyan]Fetched {len(collected_data)} valid pages. Saving files...", highlight=False)
        save_final_output(collected_data, output_dir, current_date, settings, nlp, progress)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        console.log("\n[red]Interrupted by user.", highlight=False)