import re
import aiohttp
from bs4 import BeautifulSoup
from better_profanity import profanity


async def fetch_4chan_comments(url, purge_bad_words=True):
    """

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

