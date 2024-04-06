import praw
from typing import List, Optional


class RedditSubmissionCommentsFetcher:

    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None, user_agent: Optional[str] = None) -> None:
        """
        Initializes the Reddit API client with the given credentials.

        :param client_id: The client ID for the Reddit application.
        :param client_secret: The client secret for the Reddit application.
        :param user_agent: A user agent string to identify the application to Reddit.
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def fetch_comments_from_url(self, post_url: str) -> List[str]:
        """
        Fetches and returns all comments from a specified Reddit post URL.

        :param post_url: The URL of the Reddit post from which to fetch comments.
        :return: A list of comments as strings.
        """
        submission = self.reddit.submission(url=post_url)
        submission.comments.replace_more(limit=None)  # Flattens the comment forest
        comments = [comment.body for comment in submission.comments.list()]
        return comments

