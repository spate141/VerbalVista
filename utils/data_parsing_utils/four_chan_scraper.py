import re
import aiohttp
from typing import List
from bs4 import BeautifulSoup
from better_profanity import profanity


async def fetch_4chan_comments(url: str, purge_bad_words: bool = True) -> List[str]:
    """
    Asynchronously fetches comments from a 4chan thread and returns a list of cleaned comments.

    This function removes post IDs and optionally censors profanity found in the comments.

    :param url: The URL of the 4chan thread to fetch comments from.
    :param purge_bad_words: Whether to censor profanity in the comments. Defaults to True.
    :return: A list of cleaned comments from the thread.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            soup = BeautifulSoup(await response.text(), 'html.parser')

    comments = soup.find_all('blockquote', class_='postMessage')

    clean_comments = []
    for comment in comments:
        text = comment.get_text()
        # Remove the post IDs using regular expressions
        text = re.sub(r'>>\d+', '', text)
        text = text.strip()
        if purge_bad_words:
            text = profanity.censor(text)
        clean_comments.append(text)
    return clean_comments

