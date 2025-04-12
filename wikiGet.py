import os
import re
import requests
import argparse
import json
from datetime import datetime
from rich.progress import Progress
from rich.console import Console

# Global Variables
WIKI_URL = "https://wiki.moltenaether.com/w/api.php"
API_KEY = None  # Provide an API key here if required (e.g., "your_api_key")
CATEGORY_BLACKLIST = [
    "Abandoned",
    "OOC",
    "Sources of Inspiration",
    "Star Trek",
    "Star Wars",
    "Talk Pages",
    "Denied or Banned",
    "Unapproved"
]
PAGE_BLACKLIST = ["User:", "Talk:", "Template:", "Special:"]
visited = set()
queue = []
total_pages = 0

console = Console(log_time=True, log_path=False)

def sanitize_filename(title: str) -> str:
    """
    Sanitize a page title to be filename-compliant by removing or replacing invalid
    characters and truncating to a maximum length of 255.
    """
    return re.sub(r'[<>:"/\\|?*]', ' ', title)[:255]


def fetch_wiki_name() -> str:
    """
    Fetch the wiki's site name from the API.

    Returns:
        str: The wiki site name, or 'UnknownWiki' if not found.
    """
    params = {
        "action": "query",
        "meta": "siteinfo",
        "siprop": "general",
        "format": "json",
    }
    if API_KEY:
        params["apikey"] = API_KEY

    try:
        response = requests.get(WIKI_URL, params=params, timeout=30)
        if response.status_code != 200:
            print(f"Error fetching wiki name: HTTP {response.status_code}")
            return "UnknownWiki"
        data = response.json()

        if "query" in data and "general" in data["query"] and "sitename" in data["query"]["general"]:
            return data["query"]["general"]["sitename"]
        else:
            print("Could not find 'sitename' in the siteinfo response.")
            return "UnknownWiki"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching wiki name: {e}")
        return "Wiki"


def fetch_all_pages() -> list:
    """
    Fetch all non-redirect pages on the wiki.

    Returns:
        list: A list of all non-redirect page titles.
    """
    global total_pages
    all_pages = []
    apcontinue = None

    params = {
        "action": "query",
        "list": "allpages",
        "format": "json",
        "aplimit": "max",
        "apfilterredir": "nonredirects"
    }

    if API_KEY:
        params["apikey"] = API_KEY

    while True:
        if apcontinue:
            params["apcontinue"] = apcontinue
        else:
            params.pop("apcontinue", None)

        try:
            response = requests.get(WIKI_URL, params=params, timeout=30)
            if response.status_code != 200:
                print(f"Error fetching all pages: HTTP {response.status_code}")
                break
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching all pages: {e}")
            break

        if "query" not in data or "allpages" not in data["query"]:
            print("Unexpected response structure when fetching all pages.")
            break

        for page in data["query"]["allpages"]:
            all_pages.append(page["title"])

        apcontinue = data.get("continue", {}).get("apcontinue")
        if not apcontinue:
            break

    total_pages = len(all_pages)
    return all_pages


def build_page_url(page_title: str) -> str:
    """
    Build the URL for a wiki page from its title.
    """
    import urllib.parse
    encoded_title = urllib.parse.quote(page_title.replace(' ', '_'))
    return f"{WIKI_URL.replace('/api.php', '')}/index.php?title={encoded_title}"


def fetch_page_data(title: str) -> dict:
    """
    Fetch the wiki text content, categories, and links for a specific page in a single call.
    Distinguishes between a failing fetch and a valid page that happens to have no text.

    Returns:
        {
          "content": <str or None>,
          "categories": <list[str]>,
          "links": <list[str]>,
          "fetch_error": <bool>
        }
    """
    data = {
        "content": None,
        "categories": [],
        "links": [],
        "fetch_error": False,
    }

    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions|categories|links",
        "rvprop": "content",
        "cllimit": "max",
        "pllimit": "max",
        "format": "json",
        "apfilterredir": "nonredirects"
    }
    if API_KEY:
        params["apikey"] = API_KEY

    try:
        response = requests.get(WIKI_URL, params=params, timeout=30)
        if response.status_code != 200:
            print(f"Error fetching page data for '{title}': HTTP {response.status_code}")
            data["fetch_error"] = True
            return data

        json_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page data for '{title}': {e}")
        data["fetch_error"] = True
        return data

    if "query" not in json_data or "pages" not in json_data["query"]:
        print(f"Unexpected structure for page '{title}' data.")
        data["fetch_error"] = True
        return data

    pages = json_data["query"]["pages"]
    for _, page_info in pages.items():
        if "revisions" in page_info:
            revision = page_info["revisions"][0]
            data["content"] = revision.get("*", "")

        if "categories" in page_info:
            data["categories"] = [cat["title"] for cat in page_info["categories"]]

        if "links" in page_info:
            raw_links = [lk["title"] for lk in page_info["links"]]
            filtered_links = []
            for l in raw_links:
                if not any(prefix.lower() in l.lower() for prefix in PAGE_BLACKLIST):
                    filtered_links.append(l)
            data["links"] = filtered_links

    return data


def crawl_wiki(
    recursive: bool,
    max_retries: int = 5,
    verbose: bool = False
) -> dict:
    """
    Crawl the wiki pages in the global queue, building a JSON structure.

    If fetch_error=True => re-queue up to max_retries times.
    If fetch_error=False => treat as successful, even if 'content' is empty
                           (some pages have categories/links but no text).

    Returns a dict:
      {
        "pages": [...],
        "categories": {...}
      }
    """

    pages_info = []
    category_map = {}
    page_retries = {}

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Processing pages...", total=total_pages)

        while queue:
            title = queue.pop(0)

            if title not in page_retries:
                page_retries[title] = 0

            # Skip if visited or blacklisted
            if title in visited or any(prefix.lower() in title.lower() for prefix in PAGE_BLACKLIST):
                if verbose:
                    progress.log(f"[yellow]Skipping page:[/] {title}", highlight=False)
                progress.update(task, advance=1)
                continue

            page_data = fetch_page_data(title)

            # If the fetch truly failed, try again up to max_retries
            if page_data["fetch_error"]:
                page_retries[title] += 1
                if page_retries[title] < max_retries:
                    if verbose:
                        progress.log(
                            f"[red]Fetch error for:[/] {title} "
                            f"(retry {page_retries[title]} of {max_retries}), re-queueing...",
                            highlight=False
                        )
                    queue.append(title)
                else:
                    if verbose:
                        progress.log(
                            f"[red]Fetch error for:[/] {title} "
                            f"({max_retries} retries exceeded), skipping...",
                            highlight=False
                        )
                progress.update(task, advance=1)
                continue

            # Mark visited
            visited.add(title)

            content = page_data["content"]
            categories = page_data["categories"]
            links = page_data["links"]

            # Check category blacklist
            blacklisted = any(
                black_cat.lower() in cat.lower()
                for black_cat in CATEGORY_BLACKLIST
                for cat in categories
            )
            if blacklisted:
                if verbose:
                    progress.log(
                        f"[yellow]Skipping page due to blacklisted category:[/] {title}",
                        highlight=False
                    )
                progress.update(task, advance=1)
                continue

            # Build page record
            page_record = {
                "title": title,
                "url": build_page_url(title),
                "content": content,
                "categories": categories,
                "links": links
            }
            pages_info.append(page_record)

            # Update category map
            for cat in categories:
                if cat not in category_map:
                    category_map[cat] = []
                category_map[cat].append(title)

            # If recursive, add links to the queue
            if recursive:
                for link_title in links:
                    if link_title not in visited and link_title not in queue and link_title != title:
                        queue.append(link_title)

            if verbose:
                progress.log(f"[green]Processing:[/] {title}", highlight=False)

            progress.update(task, advance=1)

    return {
        "pages": pages_info,
        "categories": category_map
    }


def main() -> None:
    """
    Produce a single JSON file named '[wiki_name] Data [YYYY-MM-DD].json' by default.
    Distinguishes between true fetch errors vs blank pages that have categories/links.

    Steps:
      1) Determine wiki name (override with --wiki-name if set).
      2) Fetch all non-redirect pages & crawl them, building a structured JSON.
      3) Write the JSON to a file, default named after the wiki & today's date.
      4) -v/--verbose logs each page's status; otherwise only the progress bar is shown.
    """
    parser = argparse.ArgumentParser(description="Wiki Crawler to JSON")

    parser.add_argument(
        "-o", "--output",
        help="Path for the output JSON file. If omitted, defaults to '[WikiName] Data [YYYY-MM-DD].json'.",
        default=None
    )
    parser.add_argument(
        "-w", "--wiki-name",
        help="Override the wiki name in the JSON data. By default, the script fetches from siteinfo.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="If set, log each page's download/skip message. Otherwise, only show progress bar."
    )

    args = parser.parse_args()

    global queue, visited, total_pages
    visited.clear()
    queue.clear()

    # 1) Wiki name
    if args.wiki_name:
        wiki_name = args.wiki_name
    else:
        wiki_name = fetch_wiki_name()

    # 2) Build default output name if none is given
    if not args.output:
        date_str = datetime.now().strftime("%Y-%m-%d")
        safe_wiki_name = sanitize_filename(wiki_name)
        args.output = f"{safe_wiki_name} Data {date_str}.json"

    # 3) Fetch & crawl pages
    queue = fetch_all_pages()
    data = crawl_wiki(recursive=True, max_retries=5, verbose=args.verbose)
    data["wiki_name"] = wiki_name

    # 4) Write JSON
    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir) and output_dir != '':
        print(f"Output folder '{output_dir}' does not exist.")
        exit(1)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Data saved to JSON file: {output_path}")
    except OSError as e:
        print(f"Error writing to file '{output_path}': {e}")


if __name__ == "__main__":
    main()
