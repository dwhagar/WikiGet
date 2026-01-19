# ────────────────────────── Standard library ──────────────────────────
import argparse
import os
import re
import time
import urllib.parse
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Set
from collections import Counter, defaultdict

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

# Global Configuration
WIKI_URL = "https://wiki.moltenaether.com/w/api.php"
API_KEY = ""

# 2000 chars is approx 1 full page double-spaced
LARGE_PAGE_THRESHOLD = 2000

# Groups smaller than this will be merged into Miscellaneous
MIN_GROUP_SIZE = 5

CATEGORY_BLACKLIST = [
    "Abandoned", "OOC", "Sources of Inspiration", "Star Trek", "Star Wars",
    "Talk Pages", "Denied or Banned", "Unapproved"
]
PAGE_BLACKLIST = ["User:", "Talk:", "Template:", "Special:", "Main Page", "Category:", "File:"]

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


# --- Utilities ---

def sanitize_filename(title: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*]', ' ', title)
    return clean.strip()[:200]


def build_page_url(title: str) -> str:
    base = WIKI_URL.replace('/api.php', '')
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


def get_preview(text: str, limit: int = 250) -> str:
    """Extracts first 'limit' chars, ignoring Markdown tables/headers."""
    lines = [line for line in text.split('\n') if not line.strip().startswith(('|', '#', '-', '*', '>'))]
    clean_text = ' '.join(lines)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    if len(clean_text) > limit:
        return clean_text[:limit] + "..."
    return clean_text


def fetch_all_pages() -> List[str]:
    pages = []
    params = {
        "action": "query", "list": "allpages", "format": "json",
        "aplimit": "max", "apfilterredir": "nonredirects"
    }
    if API_KEY: params["apikey"] = API_KEY
    apcontinue = None
    while True:
        if apcontinue:
            params["apcontinue"] = apcontinue
        else:
            params.pop("apcontinue", None)
        try:
            response = requests.get(WIKI_URL, params=params, timeout=30).json()
        except requests.RequestException:
            break
        pages.extend(p["title"] for p in response["query"]["allpages"])
        apcontinue = response.get("continue", {}).get("apcontinue")
        if not apcontinue: break
    return pages


# --- Core Logic ---

def fetch_page_content(title: str, verbose: bool) -> tuple[str, Optional[dict], Optional[str]]:
    if verbose:
        console.log(f"[dim]Fetching: {title}[/dim]", highlight=False)

    if any(prefix in title for prefix in PAGE_BLACKLIST):
        return "blacklisted", None, None

    params = {
        "action": "parse", "page": title, "prop": "text|categories",
        "format": "json", "formatversion": 2,
    }
    if API_KEY: params["apikey"] = API_KEY

    # Retry Configuration
    max_retries = 5
    base_delay = 1.0

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(WIKI_URL, params=params, timeout=30)

            # Handle Rate Limits (429) and Server Errors (5xx)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempt < max_retries:
                    sleep_time = base_delay * (2 ** attempt)
                    if verbose:
                        console.log(
                            f"[yellow]Rate limited/Server error ({resp.status_code}) on '{title}'. Retrying in {sleep_time}s...[/]",
                            highlight=False)
                    time.sleep(sleep_time)
                    continue
                else:
                    return "error", None, f"Max retries exceeded (Status {resp.status_code})"

            resp.raise_for_status()
            data = resp.json()
            break

        except requests.RequestException as e:
            if attempt < max_retries:
                sleep_time = base_delay * (2 ** attempt)
                if verbose:
                    console.log(f"[yellow]Network error on '{title}': {e}. Retrying in {sleep_time}s...[/]",
                                highlight=False)
                time.sleep(sleep_time)
                continue
            else:
                return "error", None, f"Network error after retries: {str(e)}"

    if "error" in data:
        return "error", None, f"API Error: {data['error'].get('info')}"

    parse = data.get("parse", {})
    if not parse:
        return "error", None, "No parse data found"

    raw_cats = [c.get("category", "") for c in parse.get("categories", [])]
    if any(bc.lower() in c.lower() for bc in CATEGORY_BLACKLIST for c in raw_cats):
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
        "url": build_page_url(title)
    }, None


def group_and_save_pages(pages_data: List[dict], output_dir: Path, date_str: str, verbose: bool):
    """
    Groups small pages, consolidates tiny groups, saves large pages, and builds an index.
    """

    # --- 1. Split into Small vs Large ---
    small_pages = []
    large_pages = []

    for page in pages_data:
        if len(page['content']) > LARGE_PAGE_THRESHOLD:
            large_pages.append(page)
        else:
            small_pages.append(page)

    console.log(
        f"[cyan]Analysis:[/]\n - Large Pages (> {LARGE_PAGE_THRESHOLD} chars): {len(large_pages)}\n - Small Pages: {len(small_pages)}",
        highlight=False)

    total_files_created = 0
    file_map = {}
    category_map = defaultdict(list)
    page_previews = {}

    # --- 2. Save Large Pages ---
    console.log("[cyan]Saving large pages individually...", highlight=False)
    for p in large_pages:
        safe_name = sanitize_filename(p['title'])
        filename = f"{safe_name}.md"
        filepath = output_dir / filename

        file_map[filename] = [p['title']]
        page_previews[p['title']] = get_preview(p['content'])
        for cat in p['categories']:
            category_map[cat].append(p['title'])

        if verbose:
            console.log(f"  [green]Saving Large Page:[/] {filename}", highlight=False)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {p['title']}\n")
                f.write(f"**Original URL:** {p['url']}\n")
                f.write(f"**Retrieved:** {date_str}\n\n")
                f.write(p['content'])
            total_files_created += 1
        except IOError as e:
            console.log(f"[red]Failed to write {safe_name}: {e}", highlight=False)

    # --- 3. Initial Grouping of Small Pages ---
    category_counts = Counter()
    for page in small_pages:
        for cat in page['categories']:
            category_counts[cat] += 1

    # Temporary holder for the first pass
    initial_groups = defaultdict(list)
    misc_bucket = []

    for page in small_pages:
        # Indexing metadata
        page_previews[page['title']] = get_preview(page['content'])
        for cat in page['categories']:
            category_map[cat].append(page['title'])

        page_cats = page['categories']
        if not page_cats:
            misc_bucket.append(page)
            continue

        best_category = sorted(page_cats, key=lambda c: (-category_counts[c], c))[0]
        initial_groups[best_category].append(page)

    # --- 4. Consolidate Tiny Groups ---
    final_groups = {}

    for category, pages in initial_groups.items():
        if len(pages) < MIN_GROUP_SIZE:
            # Group is too small -> Move to Misc
            misc_bucket.extend(pages)
        else:
            # Group is big enough -> Keep it
            final_groups[category] = pages

    # --- 5. Save Grouped Pages ---
    console.log(
        f"[cyan]Saving {len(small_pages)} small pages into {len(final_groups) + (1 if misc_bucket else 0)} files...",
        highlight=False)

    def write_bucket(filename_base, pages_list):
        safe_name = sanitize_filename(filename_base)
        if not safe_name.strip():
            safe_name = "Uncategorized"

        filename = f"{safe_name}.md"
        final_path = output_dir / filename

        file_map[filename] = [p['title'] for p in pages_list]

        if verbose:
            console.log(f"  [blue]Saving Group ({len(pages_list)} pages):[/] {filename}", highlight=False)
            for p in pages_list:
                console.log(f"    - {p['title']}", highlight=False)

        try:
            with open(final_path, "w", encoding="utf-8") as f:
                f.write(f"# Category: {filename_base}\n\n")
                f.write(f"*Collection of {len(pages_list)} pages retrieved on {date_str}*\n\n")

                f.write("## Table of Contents\n")
                for p in pages_list:
                    anchor = re.sub(r'[^a-zA-Z0-9\s-]', '', p['title']).replace(' ', '-').lower()
                    f.write(f"- [{p['title']}](#{anchor})\n")
                f.write("\n---\n\n")

                for p in pages_list:
                    anchor = re.sub(r'[^a-zA-Z0-9\s-]', '', p['title']).replace(' ', '-').lower()
                    f.write(f"<a name='{anchor}'></a>\n")
                    f.write(f"# {p['title']}\n")
                    f.write(f"**Original URL:** {p['url']}\n\n")
                    f.write(p['content'])
                    f.write("\n\n---\n\n")
            return True
        except IOError as e:
            console.log(f"[red]Failed to write {safe_name}: {e}", highlight=False)
            return False

    # Save the valid groups
    for category, pages in final_groups.items():
        if write_bucket(category, pages):
            total_files_created += 1

    # Save the consolidated Miscellaneous bucket
    if misc_bucket:
        # Sort misc bucket for cleaner file
        misc_bucket.sort(key=lambda x: x['title'])
        if write_bucket("Miscellaneous_Pages", misc_bucket):
            total_files_created += 1

    # --- 6. Generate Master Index ---
    console.log("[cyan]Generating Master Index...", highlight=False)
    index_path = output_dir / "_Index.md"
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(f"# Wiki Master Index\n")
            f.write(f"**Generated:** {date_str}\n")
            f.write(f"**Total Files:** {total_files_created}\n")
            f.write(f"**Total Pages:** {len(pages_data)}\n\n")

            f.write("## 1. Category Cross-Reference\n")
            f.write("List of all categories and the pages contained within them.\n\n")
            for cat in sorted(category_map.keys()):
                f.write(f"### {cat}\n")
                for title in sorted(category_map[cat]):
                    f.write(f"- {title}\n")
                f.write("\n")

            f.write("---\n\n")

            f.write("## 2. File Listing & Page Previews\n")
            # Sort files alphabetically
            for filename in sorted(file_map.keys()):
                f.write(f"### File: `{filename}`\n")
                pages = sorted(file_map[filename])
                for title in pages:
                    preview = page_previews.get(title, "No preview available.")
                    f.write(f"#### {title}\n")
                    f.write(f"> {preview}\n\n")
                f.write("\n")

        console.log(f"[green]Index saved to: {index_path}", highlight=False)
    except IOError as e:
        console.log(f"[red]Failed to write Index: {e}", highlight=False)

    # Final Warning
    console.log(f"[green]Operation Complete. Created {total_files_created} content files.", highlight=False)
    if total_files_created > 500:
        console.log(f"[red bold]WARNING: {total_files_created} files created. Exceeds 500 source limit!",
                    highlight=False)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Wiki to Markdown Grouper")
    parser.add_argument("-o", "--output", default=".", help="Output directory")
    parser.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 1) * 4))
    parser.add_argument("--test-run", action="store_true", help="Process only first 20 pages")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed errors and processing steps")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d")

    console.log("[cyan]Fetching page list...", highlight=False)
    pages = fetch_all_pages()
    if args.test_run:
        pages = pages[:20]

    collected_data = []

    with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            EMATimeRemainingColumn(alpha=0.2),
            console=console,
            refresh_per_second=4,
    ) as progress:
        task_id = progress.add_task("Fetching content...", total=len(pages))

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_title = {
                executor.submit(fetch_page_content, title, args.verbose): title for title in pages
            }

            for future in concurrent.futures.as_completed(future_to_title):
                title = future_to_title[future]
                try:
                    status, data, error = future.result()
                    if status == "success":
                        collected_data.append(data)
                    elif status == "error" and args.verbose:
                        progress.console.print(f"[red]Error {title}:[/] {error}", highlight=False)
                except Exception as e:
                    progress.console.print(f"[red]Critical {title}:[/] {e}", highlight=False)

                progress.advance(task_id)

    console.log(f"[cyan]Fetched {len(collected_data)} valid pages. Grouping and saving...", highlight=False)
    group_and_save_pages(collected_data, output_dir, current_date, args.verbose)


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        console.log("[red]Interrupted.", highlight=False)