import os
from typing import List, Dict
from .reddit_comment_parser import RedditSubmissionCommentsFetcher
from .hacker_news_scraper import scrape_hn_comments
from .four_chan_scraper import fetch_4chan_comments
from .youtube_scraper import scrape_youtube_video_transcript


def write_data_to_file(
        document_dir: str = None, full_documents: List[Dict[str, str]] = None, single_file_flag: bool = False
):
    """
    Write various full documents from `full_documents` to document_dir.
    :param document_dir: "data/past_surveys/"
    :param full_documents: [{"file_name": "file_name", "full_document": "full_document"}]
    :param single_file_flag: Save as single file or multiple files flag.
    :return:
    """
    if single_file_flag:
        directory_name = '_+_'.join([doc['file_name'].replace(' ', "_")[:15] for doc in full_documents])
        full_directory_path = os.path.join(document_dir, directory_name)
        # Ensure the directory exists
        os.makedirs(full_directory_path, exist_ok=True)
        # Step 2: Create the file name and full text
        file_name = directory_name + ".data.txt"
        full_file_path = os.path.join(full_directory_path, file_name)
        full_text = ''
        for i in full_documents:
            full_text = full_text + f"""```\nFILE TITLE: {i['file_name']}\nCONTENT: {i['full_document']}\n```""" + '\n\n\n'
        full_text = full_text.strip()
        # Write the full text to the file
        with open(full_file_path, 'w') as file:
            file.write(full_text)
        return [directory_name]
    else:
        directory_names = [doc['file_name'].replace(' ', "_") for doc in full_documents]
        full_texts = [doc['full_document'] for doc in full_documents]
        for directory_name, full_text in zip(directory_names, full_texts):
            full_directory_path = os.path.join(document_dir, directory_name)
            os.makedirs(full_directory_path, exist_ok=True)
            file_name = directory_name + ".data.txt"
            full_file_path = os.path.join(full_directory_path, file_name)
            with open(full_file_path, 'w') as file:
                file.write(full_text)
        return directory_names
