from langchain.document_loaders import SeleniumURLLoader
from urllib.parse import urlparse

from utils import log_info
from utils.data_parsing_utils import scrape_hn_comments, fetch_4chan_comments, scrape_youtube_video_transcript


def is_youtube_url(url):
    """Check if the URL is a valid YouTube URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    return domain in ['www.youtube.com', 'youtube.com', 'youtu.be']


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


def parse_url(url, msg, return_data=False):
    """

    :param url: URL
    :param msg: Streamlit toast message object
    :param return_data:
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
        loader = SeleniumURLLoader(urls=[url], browser='chrome', headless=True)
        if return_data:
            data = loader.load()
            return data
        data = loader.load()[0]
        text = data.page_content
    return text
