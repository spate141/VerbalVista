from selenium import webdriver
from urllib.parse import urlparse
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from utils import log_info
from utils.data_parsing_utils import scrape_hn_comments, fetch_4chan_comments, scrape_youtube_video_transcript


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


def get_webpage_text(url: str) -> str:
    """
    Fetch the text content of a webpage given its URL using Selenium.

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


def parse_url(url, msg):
    """

    :param url: URL
    :param msg: Streamlit toast message object
    :return:
    """
    if is_hacker_news_url(url):
        msg.toast(f'Processing HackerNews URL...')
        log_info('Parsing HackerNews URL')
        comments = scrape_hn_comments(url)
        text = '\n'.join(comments)

    elif is_4chan_url(url):
        msg.toast(f'Processing 4chan URL...')
        log_info('Parsing 4chan URL')
        comments = fetch_4chan_comments(url)
        text = '\n'.join(comments)

    elif is_youtube_url(url):
        msg.toast(f'Processing YouTube URL...')
        log_info('Parsing YouTube URL')
        text = scrape_youtube_video_transcript(url)

    else:
        msg.toast(f'Processing Normal URL...')
        log_info('Parsing Normal URL')
        text = get_webpage_text(url)
    return text
