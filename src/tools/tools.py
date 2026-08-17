import os
import requests
from langchain.tools import tool
from dotenv import load_dotenv
from tavily import TavilyClient
from rich import print

from DrissionPage import ChromiumPage, ChromiumOptions
from rich import print

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search_web(query: str) -> str:
    """
    Tool name: search_web
    Search the internet for a text query. Returns titles, URLs, and snippets.
    Use this tool ONLY to find relevant URLs.
    Do NOT use this to open or read a URL directly — use scrape_web for that.
    Do NOT call any tool named 'open_url', 'browse', or 'visit_url'.
    """
    search_response = tavily.search(query=query, max_results=3)

    results = []

    for q in search_response.get("results", []):
        title = q.get("title", "")
        url = q.get("url", "")
        content = q.get("content", "")[:200]

        results.append(
            f"Title: {title}\nURL: {url}\nSnippet: {content}\n"
        )
    
    return "\n----\n".join(results) if results else "Tidak ada hasil ditemukan."


@tool
def scrape_web(url: str) -> str:
    """
    Tool name: scrape_web
    Read the full text content from a specific webpage URL (article, blog, news).
    Use this tool to open and read a URL. This is the ONLY tool for visiting URLs.
    Do NOT call 'open_url', 'browse', or 'visit_url' — they do not exist.
    Always call search_web first to get a URL, then call scrape_web to read it.

    Args:
        url (str): The specific article/news/blog URL to read.
    """
    page = None
    try:
        co = ChromiumOptions()
        co.headless(True)
        co.set_argument("--no-sandbox")
        co.set_argument("--disable-gpu")
        co.set_argument("--disable-dev-shm-usage")

        page = ChromiumPage(co)
        page.get(url, timeout=15)

        # tunggu render JS (DOM stabil) sebelum ambil konten
        page.wait.doc_loaded(timeout=10)
        page.wait(1.5)  # buffer tambahan untuk konten yang lazy-load

        title = page.title or ""

        # buang script/style/noscript lewat JS langsung di DOM
        # (ChromiumElement tidak punya method .remove())
        page.run_js(
            "document.querySelectorAll('script,style,noscript,svg,iframe')"
            ".forEach(e => e.remove());"
        )

        body = page.ele("tag:body")
        content = body.text if body else page.html

        # rapikan whitespace berlebih
        content = "\n".join(
            line.strip() for line in content.splitlines() if line.strip()
        )

        # Batasi output agar tidak melebihi token limit model
        content_trimmed = content[:1500]
        return f"Title: {title}\nURL: {url}\n\n{content_trimmed}"

    except Exception as e:
        return f"Gagal scraping {url}: {str(e)}"

    finally:
        if page is not None:
            try:
                page.quit()
            except Exception:
                pass
