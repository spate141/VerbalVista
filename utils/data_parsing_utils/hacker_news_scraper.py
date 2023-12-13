import aiohttp
import asyncio
from bs4 import BeautifulSoup
from utils import log_debug, log_info


async def scrape_hn_comments(url):
    """Scrape comments from a Hacker News post, handling pagination."""
    async with aiohttp.ClientSession() as session:
        comments = await scrape_comments_from_page(session, url, 1)
    return comments


async def scrape_comments_from_page(session, url, page_number):
    """Scrape comments from a single page and follow pagination."""
    comments = []
    try:
        async with session.get(url) as response:
            soup = BeautifulSoup(await response.text(), 'html.parser')

        # Find all comment elements - Adjust the selector as per the page structure
        comment_elements = soup.select('.comment')
        for element in comment_elements:
            comment_text = element.get_text(strip=True)
            comments.append(comment_text)

        log_info(f"Page {page_number}: Scraped {len(comment_elements)} comments.")

        # Find the link to the next page of comments
        more_link = soup.find('a', string=lambda x: x and 'more comments' in x)
        if more_link and 'href' in more_link.attrs:
            next_page_url = 'https://news.ycombinator.com/' + more_link['href']
            await asyncio.sleep(1)  # Respectful delay before making the next request
            comments.extend(await scrape_comments_from_page(session, next_page_url, page_number + 1))

    except Exception as e:
        log_debug(f"Error scraping page {page_number}: {e}")

    return comments
