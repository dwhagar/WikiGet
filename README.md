# WikiGet for NotebookLM

This project contains `wikiGet.py`, a Python script designed to download all content from a MediaWiki site and consolidate it into a single Markdown file, optimized for analysis with Google's NotebookLM.

## Features

-   **Consolidated Output**: Fetches all pages from a wiki and saves them in a single `NotebookLM_Export.md` file.
-   **NotebookLM Optimized**: Formats the output with clear headers, metadata (URL, categories), and Markdown separators, making it easy for language models to parse.
-   **Configuration Driven**: All settings are managed via a `config.json` file, making the script reusable and easy to customize without editing code.
-   **Performance & Reliability**: Uses concurrent downloads to speed up the process and includes automatic retries for network errors or server rate limiting.
-   **Content Cleaning**: Converts wiki HTML to clean Markdown, stripping unnecessary elements like navigation boxes, edit links, and other boilerplate.

## Usage

The script is run from the command line:

```bash
python wikiGet.py [OPTIONS]
```

### Options

-   `-c, --config FILE`: Specifies the path to a JSON configuration file. Defaults to `config.json`.
-   `-o, --output DIRECTORY`: Sets the output directory for the `NotebookLM_Export.md` file. This overrides the setting in the config file.
-   `--workers N`: Defines the number of concurrent workers for fetching pages. This overrides the setting in the config file.
-   `--test-run`: Processes only the first 20 pages, which is useful for testing configuration and connectivity.
-   `-v, --verbose`: Enables detailed logging of the script's progress and any errors encountered.

## Configuration

The script is controlled by a `config.json` file. A `config.json.example` is provided to show the structure.

```json
{
  "wikiGet": {
    "wiki_url": "https://your-wiki.com/w/api.php",
    "api_key": "YOUR_API_KEY_IF_NEEDED",
    "category_blacklist": [
      "CategoryToSkip1",
      "CategoryToSkip2"
    ],
    "page_blacklist": [
      "User:",
      "Talk:",
      "Template:"
    ],
    "output_directory": "output",
    "workers": 8,
    "test_run": false,
    "verbose": false
  }
}
```

### Configuration Fields

-   `wiki_url`: **(Required)** The full URL to your MediaWiki `api.php` endpoint.
-   `api_key`: An API key for your wiki, if one is required for access.
-   `category_blacklist`: A list of categories to exclude from the export. Pages in these categories will be skipped.
-   `page_blacklist`: A list of page title prefixes to exclude (e.g., `User:` or `Talk:`).
-   `output_directory`: The directory where `NotebookLM_Export.md` will be saved.
-   `workers`: The number of parallel threads to use for downloading pages.
-   `test_run`: If `true`, the script will only fetch the first 20 pages.
-   `verbose`: If `true`, the script will print detailed progress to the console.
