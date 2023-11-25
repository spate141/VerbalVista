import time
import requests
from bs4 import BeautifulSoup
from utils.logging_module import log_debug, log_info


def scrape_hn_comments(url):
    """Scrape comments from a Hacker News post, handling pagination."""
    comments = scrape_comments_from_page(url, 1)
    return comments


def scrape_comments_from_page(url, page_number):
    """Scrape comments from a single page and follow pagination."""
    comments = []
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

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
        time.sleep(1)  # Respectful delay before making the next request
        # log_info(f"Moving to page {page_number + 1}")
        comments.extend(scrape_comments_from_page(next_page_url, page_number + 1))

    return comments

