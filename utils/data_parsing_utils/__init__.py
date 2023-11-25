import os
from .reddit_comment_parser import RedditSubmissionCommentsFetcher
from .hacker_news_scraper import scrape_hn_comments
from .four_chan_scraper import fetch_4chan_comments
from .youtube_scraper import scrape_youtube_video_transcript


def write_data_to_file(
        uploaded_file_name: str = None, document_dir: str = None, full_document: str = None, document_desc: str = None
):
    """
    Save the text to a file in a folder.
    :param uploaded_file_name:
    :param document_dir:
    :param full_document:
    :param document_desc:
    :return:
    """
    file_name_no_ext = os.path.splitext(uploaded_file_name)[0]
    file_name_with_ext = file_name_no_ext + '.data.txt'
    file_meta_with_ext = file_name_no_ext + '.meta.txt'
    file_dir = os.path.join(document_dir, file_name_no_ext)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    document_file_data_path = os.path.join(file_dir, file_name_with_ext)
    document_file_meta_path = os.path.join(file_dir, file_meta_with_ext)
    with open(document_file_data_path, 'w') as f:
        f.write(full_document)
    with open(document_file_meta_path, 'w') as f:
        f.write(document_desc)
    return document_file_data_path
