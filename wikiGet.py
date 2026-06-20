# ────────────────────────── Standard library ──────────────────────────
import argparse
import json
import os
import re
import time
import urllib.parse
import concurrent.futures
from collections import defaultdict
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

console = Console(log_time=True, log_path=False)

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
        "category_blacklist": [
            "Abandoned", "OOC", "Sources of Inspiration", "Star Trek", "Star Wars",
            "Talk Pages", "Denied or Banned", "Unapproved"
        ],
        "page_blacklist": ["User:", "Talk:", "Template:", "Special:", "Main Page", "Category:", "File:"],
        "output_directory": ".",
        "workers": min(32, (os.cpu_count() or 1) * 4),
        "test_run": False,
        "verbose": False
    }
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f).get("wikiGet", {})
            # Merge with defaults to ensure all keys are present
            default_config.update(config)
            console.log(f"[green]Loaded configuration from: {config_path}", highlight=False)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        console.log(f"[yellow]Could not load {config_path} ({e}), using default settings.", highlight=False)
    return default_config


# --- Utilities ---

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
    else: # This else belongs to the for loop, executing if the loop completes without break
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


def save_for_notebooklm(pages_data: List[dict], output_dir: Path, date_str: str, verbose: bool):
    """
    Saves all wiki content into a single, structured Markdown file optimized for NotebookLM.
    Includes a master index and clear page delimiters.
    """
    output_file = output_dir / "NotebookLM_Export.md"
    total_pages = len(pages_data)
    console.log(f"[cyan]Structuring {total_pages} pages for NotebookLM...", highlight=False)

    # Sort pages alphabetically by title for consistent output
    pages_data.sort(key=lambda p: p['title'])

    # --- Pre-computation for Index ---
    all_titles = [p['title'] for p in pages_data]
    category_map = defaultdict(list)
    for page in pages_data:
        for cat in page.get('categories', []):
            category_map[cat].append(page['title'])

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            # --- 1. Main Header ---
            f.write(f"# Consolidated Wiki Export for NotebookLM\n")
            f.write(f"**Generated:** {date_str}\n")
            f.write(f"**Total Pages:** {total_pages}\n\n")
            f.write("---\n\n")

            # --- 2. Master Index and Overview ---
            f.write(f"## Master Index and Overview\n\n")
            
            # --- 2a. All Pages Index ---
            f.write(f"### All Pages ({total_pages} total, alphabetical)\n")
            for title in all_titles:
                f.write(f"- {title}\n")
            f.write("\n")

            # --- 2b. Category Cross-Reference ---
            f.write(f"### Category Cross-Reference\n")
            if category_map:
                for category in sorted(category_map.keys()):
                    f.write(f"- **{category}**\n")
                    for title in sorted(category_map[category]):
                        f.write(f"  - {title}\n")
            else:
                f.write("No categories found.\n")
            f.write("\n---\n\n")

            # --- 3. Full Page Content ---
            f.write("## Full Page Content\n\n")
            for i, page in enumerate(pages_data):
                if verbose:
                    console.log(f"  [dim]Writing page {i+1}/{total_pages}: {page['title']}[/dim]", highlight=False)
                
                # Page Header
                f.write(f"### Page: {page['title']}\n\n")
                
                # Metadata
                f.write(f"**Original URL:** <{page['url']}>\n")
                categories = ", ".join(sorted(page.get('categories', [])))
                if categories:
                    f.write(f"**Categories:** {categories}\n")
                f.write("\n---\n\n")
                
                # Content
                f.write(page['content'])
                
                # Explicit Footer
                f.write(f"\n\n--- END OF PAGE: {page['title']} ---\n\n")

        console.log(f"[green]Successfully created optimized NotebookLM export at: {output_file}", highlight=False)

    except IOError as e:
        console.log(f"[red]Failed to write NotebookLM export file: {e}", highlight=False)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="A tool to download and process pages from a MediaWiki site for NotebookLM.")
    parser.add_argument("-c", "--config", default="config.json", help="Path to the configuration file (default: config.json).")
    parser.add_argument("-o", "--output", help="Output directory for the export file. Overrides config file setting.")
    parser.add_argument("--workers", type=int, help="Number of concurrent workers. Overrides config file setting.")
    parser.add_argument("--test-run", action="store_true", help="Process only first 20 pages. Overrides config file setting.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed processing steps. Overrides config file setting.")
    args = parser.parse_args()

    # Load configuration from file
    settings = load_config(args.config)

    # Override settings with command-line arguments if provided
    if args.output: settings['output_directory'] = args.output
    if args.workers: settings['workers'] = args.workers
    # Action="store_true" args need to be checked if they are True
    if args.test_run: settings['test_run'] = True
    if args.verbose: settings['verbose'] = True

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

    console.log(f"[cyan]Fetched {len(collected_data)} valid pages. Saving for NotebookLM...", highlight=False)
    save_for_notebooklm(collected_data, output_dir, current_date, settings['verbose'])


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        console.log("\n[red]Interrupted by user.", highlight=False)
