import re
import os
import tempfile
import docx2txt
from io import BytesIO
from typing import List
from pypdf import PdfReader
from urllib.parse import urlparse
from langchain.docstore.document import Document
from langchain.document_loaders import SeleniumURLLoader
from langchain.document_loaders import UnstructuredEmailLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.logging_module import log_debug, log_error, log_info
from utils.data_parsing_utils.hacker_news_scraper import scrape_hn_comments
from utils.data_parsing_utils.four_chan_scraper import fetch_4chan_comments
from utils.data_parsing_utils.youtube_scraper import scrape_youtube_video_transcript

def parse_docx(file: BytesIO) -> str:
    """
    Parse word file and return content as string of text.
    :param file: Word file
    :return:
    """
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def parse_pdf(file: BytesIO) -> str:
    """
    Parse pdf file and return content as string of text.
    :param file: PDF file
    :return:
    """
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    output = ' '.join(output)
    return output


def parse_txt(file: BytesIO) -> str:
    """
    Parse text file and return content as string of text.
    :param file: Normal text file
    :return:
    """
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def text_to_docs(text: str | List[str]) -> List[Document]:
    """
    Converts a string or list of strings to a list of Documents with metadata.
    """
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


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


def parse_email(file: BytesIO):
    """

    :param file:
    """
    # Save the file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.eml') as temp_file:
        temp_filename = temp_file.name
        log_debug(f"tmp email files saved at: {temp_filename}")
        temp_file.write(file.getvalue())

    loader = UnstructuredEmailLoader(file_path=temp_filename)
    raw_documents = loader.load()
    texts = []
    for i in raw_documents:
        texts.append(i.page_content)
    text = ' '.join(texts)

    # Remove the temporary file
    os.remove(temp_filename)
    log_debug(f"tmp email files removed from: {temp_filename}")
    return text


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

