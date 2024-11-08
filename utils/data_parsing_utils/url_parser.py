import re
import os
import shutil
import asyncio
import hashlib
from typing import Optional, Any
from selenium import webdriver
from urllib.parse import urlparse
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from utils import log_info
from utils.data_parsing_utils.hacker_news_scraper import scrape_hn_comments
from utils.data_parsing_utils.four_chan_scraper import fetch_4chan_comments
from utils.data_parsing_utils.youtube_scraper import scrape_youtube_video_transcript


def is_youtube_url(url: str) -> bool:
    """
    Checks if the given URL is a valid YouTube URL.

    :param url: URL to check.
    :return: True if the URL is a YouTube URL, False otherwise.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    return domain in ['www.youtube.com', 'youtube.com', 'youtu.be', "youtu.be", "m.youtube.com"]


def is_hacker_news_url(url: str) -> bool:
    """
    Checks if the given URL is a valid Hacker News URL.

    :param url: URL to check.
    :return: True if the URL is a Hacker News URL, False otherwise.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    return domain in ['news.ycombinator.com']


def is_4chan_url(url: str) -> bool:
    """
    Checks if the given URL is a valid 4chan URL.

    :param url: URL to check.
    :return: True if the URL is a 4chan URL, False otherwise.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    return domain in ['boards.4chan.org', 'boards.4channel.org']


async def get_webpage_text(url: str) -> str:
    """
    Asynchronously fetches the text content of a webpage given its URL using Selenium.

    :param url: URL of the webpage.
    :return: The text content of the webpage.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: fetch_page_text(url)
    )


def fetch_page_text(url: str) -> str:
    """
    Fetches the text content of a webpage given its URL using Selenium in a synchronous manner.

    :param url: URL of the webpage.
    :return: The text content of the webpage.
    """
    options = Options()
    options.add_argument("--headless")  # Run in headless mode for background execution
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-features=NetworkService")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--disable-features=VizDisplayCompositor")
    service = Service(
        executable_path=shutil.which('chromedriver'),
        log_output=os.path.join(os.getcwd(), 'selenium.log'),
    )
    # service = Service(ChromeDriverManager().install())
    with webdriver.Chrome(service=service, options=options) as driver:
        driver.get(url)
        page_text = driver.find_element(By.TAG_NAME, "body").text
        # driver.quit()
        return page_text


async def process_url(url: str, msg: Optional[Any] = None) -> str:
    """
    Processes the given URL to fetch comments or content based on the type of the URL (Hacker News, 4chan, YouTube, or other).

    :param url: The URL to process.
    :param msg: Optional message object for displaying toast notifications.
    :return: Extracted text content or comments from the URL.
    """
    if is_hacker_news_url(url):
        if msg:
            msg.toast(f'Processing HackerNews URL...')
        log_info('Parsing HackerNews URL')
        comments = await scrape_hn_comments(url)
        text = '\n'.join(comments)

    elif is_4chan_url(url):
        if msg:
            msg.toast(f'Processing 4chan URL...')
        log_info('Parsing 4chan URL')
        comments = await fetch_4chan_comments(url)
        text = '\n'.join(comments)

    elif is_youtube_url(url):
        if msg:
            msg.toast(f'Processing YouTube URL...')
        log_info('Parsing YouTube URL')
        text = await scrape_youtube_video_transcript(url)

    else:
        if msg:
            msg.toast(f'Processing Normal URL...')
        log_info('Parsing Normal URL')
        text = await get_webpage_text(url)
    return text


def url_to_filename(url: str) -> str:
    """
    Converts a URL to a sanitized filename, incorporating parts of the domain and path, and a short hash for uniqueness.

    :param url: The URL to convert.
    :return: A sanitized filename string.
    """
    # Parse the URL
    parsed_url = urlparse(url)

    # Extract the domain
    domain = parsed_url.netloc.split('.')[-2]  # Get the second last part of the domain

    # Extract the last segment of the path
    last_segment = parsed_url.path.split('/')
    try:
        last_segment = [i for i in last_segment if i][-1]
    except:
        last_segment = parsed_url.netloc
    short_segment = re.sub(r'[^a-zA-Z0-9]', '_', last_segment)  # Keep first 10 chars

    # Generate a short hash for uniqueness
    url_hash = hashlib.md5(url.encode()).hexdigest()[:4]  # Short hash for uniqueness

    # Combine to form the filename
    filename = f"{domain}_{short_segment}_{url_hash}.txt"

    return filename

