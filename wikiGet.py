import requests
import time
import argparse

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

def fetch_page_content(title):
    """
    Fetch the wikitext content of a given page, following redirects if necessary.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions|categories",  # Fetch content and categories
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

    print(f"Processing: {title}")

    for page_id, page_info in pages.items():
        if "revisions" in page_info:
            content = page_info["revisions"][0]["*"]  # Page content (wikitext)

            # Check for redirect (i.e., #REDIRECT at the start of the page)
            if content.startswith("#REDIRECT"):
                # Extract the redirect target from the content
                redirect_target = content.split("\n")[0].replace("#REDIRECT", "").strip()
                if redirect_target[:2] == "[[":
                    redirect_target = redirect_target[2:]
                elif redirect_target[:3] == ":[[":
                    redirect_target = redirect_target[3:]
                if redirect_target[0] == ":":
                    redirect_target = redirect_target[1:]
                if redirect_target[-2:] == "]]":
                    redirect_target = redirect_target[:-2]
                redirect_target = redirect_target.split('#')[0]

                print(f"Redirected from {title} to {redirect_target}")

                # Add the redirect target to the queue if not already visited or in the queue
                if redirect_target not in visited and redirect_target not in queue:
                    queue.append(redirect_target)

                return None  # Skip fetching the redirected content immediately

            # Check categories for blacklist
            if "categories" in page_info:
                categories = [cat["title"] for cat in page_info["categories"]]
                for blacklist in CATEGORY_BLACKLIST:
                    if any(blacklist.lower() in category.lower() for category in categories):
                        print(f"Skipping page {title} due to blacklisted category: '{blacklist}'")
                        return None

            return content  # Return the page content

    return None  # Return None if page has no content

def crawl_wiki(output_file):
    """
    Crawl the wiki starting from the global queue of page titles and write content to a file.
    """
    while queue:
        title = queue.pop(0)
        if title in visited:
            print(f"Skipping already visited page: {title}")
            continue
        visited.add(title)

        # Fetch content
        content = fetch_page_content(title)

        # If content exists, write it to the file
        if content:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"= {title} =\n")
                f.write(content)
                f.write("\n* * *\n")

        # Fetch linked pages and add to queue
        links = fetch_page_links(title)
        for link in links:
            if link not in visited and link not in queue:
                queue.append(link)

        # Respect API limits
        time.sleep(0.5)

def fetch_page_links(title):
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
                for link in page_info["links"]:
                    for blacklist in PAGE_BLACKLIST:
                        if blacklist.lower() in link['title'].lower():
                            if not link['title'] in visited:
                                print(f"Skipping page {link['title']} due to blacklisted page: '{blacklist}'")
                                visited.add(link['title'])
                        else:
                            links.append(link["title"])

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

    # Initialize global queue with starting category pages
    global queue
    queue = fetch_category_pages(args.category)

    # Crawl the wiki and save results
    crawl_wiki(args.output)

    print(f"Data saved to {args.output}")

if __name__ == "__main__":
    main()
