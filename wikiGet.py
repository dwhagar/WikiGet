import os  # Import os for directory-related checks
import requests
import time
import argparse
import re  # Import re for sanitizing filenames
from tqdm import tqdm  # Import tqdm for progress bar

# Global Variables
WIKI_URL = "https://wiki.moltenaether.com/w/api.php"
API_KEY = None  # Optional: Provide an API key here if required (e.g., "your_api_key")
CATEGORY_BLACKLIST = [
    "Abandoned",
    "OOC",
    "Stubs",
    "Disambiguation",
    "Sources of Inspiration",
    "Star Trek",
    "Star Wars"
]  # Add blacklist categories here
PAGE_BLACKLIST = ["User:"]
visited = set()
queue = []
total_pages = 0
processed_pages = 0


def sanitize_filename(title):
    """
    Sanitize a title to be file name compliant by removing or replacing invalid characters.
    """
    return re.sub(r'[<>:"/\\|?*]', ' ', title)[:255]


def validate_output_folder(output_folder):
    """
    Validate that the specified output folder exists and is writable.
    """
    if not os.path.exists(output_folder):
        print(f"The specified folder '{output_folder}' does not exist.")
        exit(1)
    if not os.access(output_folder, os.W_OK):
        print(f"The specified folder '{output_folder}' is not writable.")
        exit(1)


def fetch_category_pages(category):
    """Fetch all pages in a given category."""
    pages = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "format": "json",
            "cmlimit": "max",
            "apfilterredir": "nonredirects",  # Ignore redirect pages
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        if API_KEY:
            params["apikey"] = API_KEY

        response = requests.get(WIKI_URL, params=params).json()
        if "query" in response:
            pages.extend(member["title"] for member in response["query"]["categorymembers"])

        cmcontinue = response.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    return pages


def fetch_all_pages():
    """Fetch all non-redirect pages on the wiki."""
    global total_pages
    all_pages = []
    apcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "format": "json",
            "aplimit": "max",
            "apfilterredir": "nonredirects",  # Ignore redirect pages
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        if API_KEY:
            params["apikey"] = API_KEY

        response = requests.get(WIKI_URL, params=params).json()
        pages = response["query"]["allpages"]

        for page in pages:
            title = page["title"]
            all_pages.append(title)

        apcontinue = response.get("continue", {}).get("apcontinue")
        if not apcontinue:
            break

    total_pages = len(all_pages)
    return all_pages


def fetch_page_categories(title):
    """Fetch categories of a given page."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "categories",
        "format": "json",
    }
    response = requests.get(WIKI_URL, params=params).json()
    pages = response["query"]["pages"]
    categories = []
    for page_id, page_info in pages.items():
        if "categories" in page_info:
            categories = [cat["title"] for cat in page_info["categories"]]
    return categories


def fetch_page_content(title):
    """Fetch the wikitext content of a given page."""
    global processed_pages
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "format": "json",
        "apfilterredir": "nonredirects",  # Ignore redirect pages
    }
    if API_KEY:
        params["apikey"] = API_KEY

    response = requests.get(WIKI_URL, params=params).json()
    pages = response["query"]["pages"]

    for page_id, page_info in pages.items():
        if "revisions" in page_info:
            content = page_info["revisions"][0]["*"]
            processed_pages += 1
            return f"= {title} =\n\n{content}"
    return None


def fetch_page_links(title):
    """Fetch all linked pages on a given page."""
    links = []
    plcontinue = None

    while True:
        params = {
            "action": "query",
            "titles": title,
            "prop": "links",
            "format": "json",
            "pllimit": "max",
        }
        if plcontinue:
            params["plcontinue"] = plcontinue
        if API_KEY:
            params["apikey"] = API_KEY

        response = requests.get(WIKI_URL, params=params).json()
        pages = response["query"]["pages"]

        for page_id, page_info in pages.items():
            if "links" in page_info:
                for link in page_info["links"]:
                    link_title = link["title"]
                    if not any(blacklist.lower() in link_title.lower() for blacklist in PAGE_BLACKLIST):
                        links.append(link_title)

        plcontinue = response.get("continue", {}).get("plcontinue")
        if not plcontinue:
            break

    return links


def crawl_wiki(output_folder, recursive, no_overwrite):
    """Crawl the wiki and save pages."""
    global processed_pages
    progress_bar = tqdm(total=total_pages, desc="Processing Pages", unit="page")

    while queue:
        title = queue.pop(0)
        sanitized_title = sanitize_filename(title)
        file_name = os.path.join(output_folder, f"{sanitized_title}.txt")

        if no_overwrite and os.path.exists(file_name):
            tqdm.write(f"Skipping existing file: {title}")  # Ensures message appears on a new line
            progress_bar.update(1)
            continue

        if title in visited or any(blacklist.lower() in title.lower() for blacklist in PAGE_BLACKLIST):
            tqdm.write(f"Skipping page: {title}")  # Message appears on a new line
            progress_bar.update(1)
            continue
        visited.add(title)

        categories = fetch_page_categories(title)
        if any(blacklist.lower() in category.lower() for blacklist in CATEGORY_BLACKLIST for category in categories):
            tqdm.write(f"Skipping page {title} due to blacklisted category.")  # New line message
            progress_bar.update(1)
            continue

        tqdm.write(f"Processing: {title}")  # Shows page title without overwriting the progress bar
        content = fetch_page_content(title)
        if content:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(content)

            # If recursive, fetch links from the page and add them to the queue
            if recursive:
                links = fetch_page_links(title)
                for link in links:
                    if link not in visited and link not in queue:
                        queue.append(link)

            time.sleep(0.5)  # Respect API limits if recursive

        progress_bar.update(1)

    progress_bar.close()


def main():
    parser = argparse.ArgumentParser(description="Wiki Crawler Script")
    parser.add_argument(
        "-c", "--category", help="The category to crawl. If omitted, will fetch all pages.",
    )
    parser.add_argument(
        "-o", "--output", default=os.getcwd(), help="Output folder to store files (default: current working directory).",
    )
    parser.add_argument(
        "-n", "--no-overwrite", action="store_true", help="Skip downloading pages that already exist in the output folder.",
    )

    args = parser.parse_args()

    # Validate output folder
    output_folder = args.output
    validate_output_folder(output_folder)

    # Initialize queue based on input
    global queue
    recursive = True
    if args.category:
        queue = fetch_category_pages(args.category)
    else:
        print("No category specified. Fetching all pages.")
        queue = fetch_all_pages()
        recursive = False  # No need to crawl links if fetching all pages

    # Crawl the wiki
    crawl_wiki(output_folder, recursive, args.no_overwrite)

    print(f"Data saved to files in '{output_folder}'.")

if __name__ == "__main__":
    main()
