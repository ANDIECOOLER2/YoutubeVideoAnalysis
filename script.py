""Scrape Insider Intelligence articles and embed them into a Chroma database."""
 
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "html2text",
#     "lxml",
#     "requests",
#     "tiktoken",
#     "tqdm",
#     "langchain",
#     "langchain-openai",
#     "langchain-text-splitters",
#     "langchain-chroma",
# ]
# ///
 
import hashlib
import html2text
import json
import lxml.html
import os
import requests
import tiktoken
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from tqdm import tqdm
 
# Cache OpenAI embeddings at .embeddings
file_store = LocalFileStore('.embeddings')
base = OpenAIEmbeddings(model="text-embedding-3-small")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(base, file_store, namespace=base.model)
 
# Chunk text into a reasonable number of tokens with some overlap
tokenizer = tiktoken.get_encoding("cl100k_base")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=lambda s: len(tokenizer.encode(s)),
    is_separator_regex=False,
)
 
 
def cached_get(url):
    """Fetch the URL, cached"""
    filename = hashlib.md5(url.encode("utf-8")).hexdigest()  # noqa: S324 non-security
    path = f".cache/{filename}"
    if not os.path.exists(path):
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            f.write(response.text)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
 
 
def index_crawl():
    """Crawl the Insider Intelligence article archive and return the URLs of the articles."""
    if os.path.exists('.urls.txt'):
        with open('.urls.txt', 'r', encoding='utf-8') as f:
            return f.read().split('\n')
 
    base = "https://www.insiderintelligence.com/articles/archive"
    urls = set()
    for page in tqdm(range(1, 1000)):
        url_count = len(urls)
        url = f"{base}/{page}"
        try:
            result = cached_get(url)
        except requests.exceptions.HTTPError:
            break
        html = lxml.html.fromstring(result)
        for a in html.xpath("//a"):
            href = a.get("href")
            if href and "/content/" in href:
                urls.add(href)
        # Exit when we stop finding new URLs
        if len(urls) == url_count:
            break
    # Persist the URLs
    with open("urls.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(urls))
    return urls
 
 
def scrape(urls):
    """Scrape the articles and return the text and metadata."""
    if os.path.exists('.scraped.json'):
        with open('.scraped.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True
    text_maker.ignore_images = True
    # Consider .images_to_alt, .ignore_emphasis
    docs = []
    for url in tqdm(urls):
        key = url.split('/')[-1]
        try:
            result = cached_get(url)
        except requests.exceptions.HTTPError:
            continue
        tree = lxml.html.fromstring(result)
        # Extract the text from the article
        result = '\n\n'.join(
            text_maker.handle(lxml.html.tostring(content).decode('utf-8')).strip()
            for content in tree.xpath("//div[@class='cb-widget-row_content']")
        )
        h1 = tree.cssselect('h1.page-title_title')[0].text_content().strip()
        docs.append({"text": result, "key": key, "h1": h1})
    # Persist the scraped data
    with open(".scraped.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)
    return docs
 
 
def embed(docs):
    """Embed the documents and persist the Chroma database."""
    metadata = [{"key": row["key"], "h1": row["h1"]} for row in docs]
    documents = text_splitter.create_documents([doc["text"] for doc in docs], metadata)
    return Chroma.from_documents(
        documents,
        cached_embedder,
        persist_directory='.chromadb',
        collection_name=os.path.basename(os.path.dirname(__file__)),
    )
 
 
if __name__ == '__main__':
    urls = index_crawl()
    docs = scrape(urls)
    embed(docs)
 
