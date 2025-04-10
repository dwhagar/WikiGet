# Answer to your question: The script is already well structured and modularized. If you wish to expand functionality,
# you could break out a separate function for page validation or combine fetch operations in one helper, but in general,
# everything is adequately broken down.

import os  # OS for directory checks
import requests
import argparse
import re  # For sanitizing filenames
from tqdm import tqdm  # Progress bar

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
]  # Add blacklist categories here
PAGE_BLACKLIST = ["User:", "Talk:", "Template:", "Special:"]
visited = set()
queue = []
total_pages = 0
processed_pages = 0


def sanitize_filename(title: str) -> str:
    """Sanitize a page title to be filename-compliant.

    Replaces or removes invalid characters and truncates to a maximum length of 255.

    Args:
        title (str): The page title to sanitize.

    Returns:
        str: A sanitized version of the title suitable for use as a filename.
    """
    # Replace invalid filename characters with a space and limit length to 255
    return re.sub(r'[<>:"/\\|?*]', ' ', title)[:255]


def validate_output_folder(output_folder: str) -> None:
    """Validate that the specified output folder exists and is writable.

    Args:
        output_folder (str): The path to the output folder.

    Raises:
        SystemExit: If the folder does not exist or is not writable.
    """
    if not os.path.exists(output_folder):
        print(f"The specified folder '{output_folder}' does not exist.")
        exit(1)
    if not os.access(output_folder, os.W_OK):
        print(f"The specified folder '{output_folder}' is not writable.")
        exit(1)


def fetch_category_pages(category: str) -> list:
    """Fetch all pages under a specified wiki category.

    Args:
        category (str): The name of the category to crawl.

    Returns:
        list: A list of page titles found in that category.
    """
    pages = []
    cmcontinue = None

    # Reuse params dict for efficiency
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "format": "json",
        "cmlimit": "max",
        "apfilterredir": "nonredirects"  # Ignore redirect pages
    }

    if API_KEY:
        params["apikey"] = API_KEY

    while True:
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        else:
            params.pop("cmcontinue", None)

        response = requests.get(WIKI_URL, params=params).json()
        if "query" in response:
            pages.extend(member["title"] for member in response["query"]["categorymembers"])

        cmcontinue = response.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    return pages


def fetch_all_pages() -> list:
    """Fetch all non-redirect pages on the wiki.

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
        "apfilterredir": "nonredirects"  # Ignore redirect pages
    }

    if API_KEY:
        params["apikey"] = API_KEY

    while True:
        if apcontinue:
            params["apcontinue"] = apcontinue
        else:
            params.pop("apcontinue", None)

        response = requests.get(WIKI_URL, params=params).json()
        pages = response["query"]["allpages"]

        for page in pages:
            all_pages.append(page["title"])

        apcontinue = response.get("continue", {}).get("apcontinue")
        if not apcontinue:
            break

    total_pages = len(all_pages)
    return all_pages


def fetch_page_categories(title: str) -> list:
    """Fetch categories for a specific wiki page.

    Args:
        title (str): The title of the wiki page.

    Returns:
        list: A list of category titles for the given page.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "categories",
        "format": "json",
    }
    if API_KEY:
        params["apikey"] = API_KEY

    response = requests.get(WIKI_URL, params=params).json()
    pages_data = response.get("query", {}).get("pages", {})
    categories = []

    for _, page_info in pages_data.items():
        if "categories" in page_info:
            categories = [cat["title"] for cat in page_info["categories"]]

    return categories


def fetch_page_content(title: str) -> str:
    """Fetch the wikitext content of a given wiki page.

    Args:
        title (str): The title of the wiki page.

    Returns:
        str: The page's content wrapped with a title header, or None if not found.
    """
    global processed_pages
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "format": "json",
        "apfilterredir": "nonredirects"  # Ignore redirect pages
    }
    if API_KEY:
        params["apikey"] = API_KEY

    response = requests.get(WIKI_URL, params=params).json()
    pages_data = response.get("query", {}).get("pages", {})

    for _, page_info in pages_data.items():
        if "revisions" in page_info:
            content = page_info["revisions"][0].get("*")
            processed_pages += 1
            return f"= {title} =\n\n{content}"

    return None


def fetch_page_links(title: str) -> list:
    """Fetch all wiki page links from a specific page.

    Args:
        title (str): The title of the wiki page.

    Returns:
        list: A list of linked page titles.
    """
    links = []
    plcontinue = None

    params = {
        "action": "query",
        "titles": title,
        "prop": "links",
        "format": "json",
        "pllimit": "max",
    }
    if API_KEY:
        params["apikey"] = API_KEY

    while True:
        if plcontinue:
            params["plcontinue"] = plcontinue
        else:
            params.pop("plcontinue", None)

        response = requests.get(WIKI_URL, params=params).json()
        pages_data = response.get("query", {}).get("pages", {})

        for _, page_info in pages_data.items():
            if "links" in page_info:
                for link in page_info["links"]:
                    link_title = link["title"]
                    # Skip if any blacklisted prefix is found
                    if not any(blacklist.lower() in link_title.lower() for blacklist in PAGE_BLACKLIST):
                        links.append(link_title)

        plcontinue = response.get("continue", {}).get("plcontinue")
        if not plcontinue:
            break

    return links


def crawl_wiki(output_folder: str, recursive: bool, no_overwrite: bool) -> None:
    """Crawl the wiki based on queued page titles and save each page to disk.

    Args:
        output_folder (str): Path to the folder where pages will be saved.
        recursive (bool): Whether to continue crawling via each page's links.
        no_overwrite (bool): If True, skip pages that already exist on disk.
    """
    global processed_pages
    progress_bar = tqdm(total=total_pages, desc="Processing Pages", unit="page")

    while queue:
        title = queue.pop(0)
        sanitized_title = sanitize_filename(title)
        file_name = os.path.join(output_folder, f"{sanitized_title}.txt")

        # Skip if file exists and no-overwrite is true
        if no_overwrite and os.path.exists(file_name):
            tqdm.write(f"Skipping existing file: {title}")
            progress_bar.update(1)
            continue

        # Skip if already visited or blacklisted
        if title in visited or any(blacklist.lower() in title.lower() for blacklist in PAGE_BLACKLIST):
            tqdm.write(f"Skipping page: {title}")
            progress_bar.update(1)
            continue

        visited.add(title)

        # Fetch categories and skip if none or blacklisted
        categories = fetch_page_categories(title)
        if not categories:
            tqdm.write(f"Skipping page {title} due to no categories.")
            progress_bar.update(1)
            continue
        if any(blacklist.lower() in category.lower() for blacklist in CATEGORY_BLACKLIST for category in categories):
            tqdm.write(f"Skipping page {title} due to blacklisted category.")
            progress_bar.update(1)
            continue

        tqdm.write(f"Processing: {title}")
        content = fetch_page_content(title)
        if content:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(content)

            if recursive:
                # If recursive, fetch links from the page and add them to the queue
                links = fetch_page_links(title)
                for link in links:
                    if link not in visited and link not in queue:
                        queue.append(link)

        progress_bar.update(1)

    progress_bar.close()


def merge_text_files(output_folder: str) -> None:
    """Merge all .txt files in the output folder into a single file named 'wiki-data.txt'.

    Args:
        output_folder (str): Path to the folder containing .txt files.
    """
    merged_file_path = os.path.join(output_folder, "wiki-data.txt")

    txt_files = [
        f for f in os.listdir(output_folder)
        if f.endswith(".txt") and f != "wiki-data.txt"
    ]

    with open(merged_file_path, "w", encoding="utf-8") as outfile:
        for txt_file in txt_files:
            file_path = os.path.join(output_folder, txt_file)
            with open(file_path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
                outfile.write("\n")


def main() -> None:
    """Parse command-line arguments, then crawl the wiki and merge text files."""
    parser = argparse.ArgumentParser(description="Wiki Crawler Script")
    parser.add_argument(
        "-c", "--category", help="The category to crawl. If omitted, fetches all pages.",
    )
    parser.add_argument(
        "-o", "--output", default=os.getcwd(),
        help="Output folder to store files (default: current working directory).",
    )
    parser.add_argument(
        "-n", "--no-overwrite", action="store_true",
        help="Skip downloading pages that already exist in the output folder.",
    )

    args = parser.parse_args()

    # Validate output folder
    output_folder = args.output
    validate_output_folder(output_folder)

    # Determine if we should be recursive based on category usage
    global queue
    recursive = True
    if args.category:
        queue = fetch_category_pages(args.category)
    else:
        print("No category specified. Fetching all pages.")
        queue = fetch_all_pages()
        recursive = False

    # Crawl the wiki
    crawl_wiki(output_folder, recursive, args.no_overwrite)

    # Merge all text files
    merge_text_files(output_folder)

    print(f"Data saved to files in '{output_folder}'. Also merged into 'wiki-data.txt'.")


if __name__ == "__main__":
    main()
