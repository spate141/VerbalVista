import asyncio
from selenium import webdriver
from urllib.parse import urlparse
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from utils import log_info
from utils.data_parsing_utils.hacker_news_scraper import scrape_hn_comments
from utils.data_parsing_utils.four_chan_scraper import fetch_4chan_comments
from utils.data_parsing_utils.youtube_scraper import scrape_youtube_video_transcript


def is_youtube_url(url):
    """Check if the URL is a valid YouTube URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    return domain in ['www.youtube.com', 'youtube.com', 'youtu.be', "youtu.be", "m.youtube.com"]


def is_hacker_news_url(url):
    """Check if the URL is a valid Hacker News URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    return domain in ['news.ycombinator.com']


def is_4chan_url(url):
    """Check if the URL is a 4chan URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    return domain in ['boards.4chan.org', 'boards.4channel.org']


async def get_webpage_text(url: str) -> str:
    """
    Fetch the text content of a webpage given its URL using Selenium.

    Args:
    url (str): URL of the webpage.

    Returns:
    str: The text content of the webpage.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: fetch_page_text(url)
    )


def fetch_page_text(url: str) -> str:
    """
    Synchronous function to fetch the text content of a webpage using Selenium.

    Args:
    url (str): URL of the webpage.

    Returns:
    str: The text content of the webpage.
    """
    options = Options()
    options.add_argument("--headless")  # Run in headless mode for background execution
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    page_text = driver.find_element(By.TAG_NAME, "body").text
    driver.quit()
    return page_text


async def process_url(url, msg=None):
    """

    :param url: URL
    :param msg: Streamlit toast message object
    :return:
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
