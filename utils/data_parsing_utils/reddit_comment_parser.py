import praw


class RedditSubmissionCommentsFetcher:
    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(client_id=client_id,
                                  client_secret=client_secret,
                                  user_agent=user_agent)

    def fetch_comments_from_url(self, post_url):
        submission = self.reddit.submission(url=post_url)
        submission.comments.replace_more(limit=None)  # Flattens the comment forest
        comments = [comment.body for comment in submission.comments.list()]
        return comments

