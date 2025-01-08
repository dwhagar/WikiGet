import requests
import time
import argparse

# Global Variables
WIKI_URL = "https://wiki.moltenaether.com/w/api.php"
API_KEY = None  # Optional: Provide an API key here if required (e.g., "your_api_key")

def fetch_category_pages(category):
    """
    Fetch all pages in a given category.
    """
    pages = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "format": "json",
            "cmcontinue": cmcontinue,
        }
        if API_KEY:
            params["apikey"] = API_KEY

        response = requests.get(WIKI_URL, params=params).json()
        pages.extend(member["title"] for member in response["query"]["categorymembers"])

        cmcontinue = response.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    return pages

def fetch_page_content(title, visited):
    """
    Fetch the wikitext content of a given page, following redirects if necessary.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",  # Fetch content only
        "rvprop": "content",
        "format": "json",
    }
    if API_KEY:
        params["apikey"] = API_KEY

    response = requests.get(WIKI_URL, params=params).json()
    pages = response["query"]["pages"]

    # Do not follow file links.
    if title.lower().startswith("file:"):
        print(f"Ignoring file link {title}.")
        return None

    for page_id, page_info in pages.items():
        if "revisions" in page_info:
            content = page_info["revisions"][0]["*"]  # Page content (wikitext)

            # Check for redirect (i.e., #REDIRECT at the start of the page)
            if content.startswith("#REDIRECT"):
                # Extract the redirect target from the content
                redirect_target = content.split("\n")[0].replace("#REDIRECT", "").strip()
                redirect_target = redirect_target.strip("[]: ").split("#")[0]

                # Check if the target is a valid page title and it's not visited
                if redirect_target not in visited:
                    visited.add(redirect_target)  # Mark the redirected page as visited
                    print(f"Redirected from {title} to {redirect_target}")
                    return fetch_page_content(redirect_target, visited)  # Recursively fetch the target page
                else:
                    print(f"Skipping already visited redirect: {redirect_target}")
                    return None  # If redirected page is already visited, skip it

            return content  # Return the page content

    return None  # Return None if page has no content

def crawl_wiki(queue, visited, output_file):
    """
    Crawl the wiki starting from a queue of page titles and write content to a file.
    """
    while queue:
        title = queue.pop(0)
        if title in visited:
            continue
        visited.add(title)

        print(f"Processing: {title}")

        # Fetch content
        content = fetch_page_content(title, visited)

        # If content exists, write it to the file
        if content:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"\n=== {title} ===\n")
                f.write(content)
                f.write("\n\n")

        # Fetch linked pages and add to queue
        links = fetch_page_links(title, visited)
        queue.extend(link for link in links if link not in visited)

        # Respect API limits
        time.sleep(0.5)

def fetch_page_links(title, visited):
    """
    Fetch all linked pages on a given page, following redirects if necessary.
    """
    links = []
    plcontinue = None

    while True:
        params = {
            "action": "query",
            "titles": title,
            "prop": "links",
            "format": "json",
            "plcontinue": plcontinue,
        }
        if API_KEY:
            params["apikey"] = API_KEY

        response = requests.get(WIKI_URL, params=params).json()
        pages = response["query"]["pages"]

        for page_id, page_info in pages.items():
            if "links" in page_info:
                links.extend(link["title"] for link in page_info["links"])

        plcontinue = response.get("continue", {}).get("plcontinue")
        if not plcontinue:
            break

    return links

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Wiki Crawler Script")
    parser.add_argument(
        "-c",
        "--category",
        required=True,
        help="The starting category to crawl (e.g., 'Example Category').",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="The name of the output plain text file (e.g., 'output.txt').",
    )

    args = parser.parse_args()

    # Initialize variables
    visited = set()

    # Get list of pages in the category
    queue = fetch_category_pages(args.category)

    # Crawl the wiki and save results
    crawl_wiki(queue, visited, args.output)

    print(f"Data saved to {args.output}")

if __name__ == "__main__":
    main()
