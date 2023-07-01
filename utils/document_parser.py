from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import Any, Dict, List
from bs4 import BeautifulSoup
from pypdf import PdfReader
import streamlit as st
from io import BytesIO
import requests
import docx2txt
import re
import os


@st.cache_data
def parse_csv(file: BytesIO) -> str:
    """
    Parse csv file and return content as text.
    :param file:
    :return:
    """
    pass


@st.cache_data
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


@st.cache_data
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


@st.cache_data
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


@st.cache_data
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


def write_text_to_file(uploaded_file_name: str = None, tmp_document_dir: str = None, full_document: str = None):
    """
    Save the text to a file in a folder.
    :param uploaded_file_name:
    :param tmp_document_dir:
    :param full_document:
    :return:
    """
    transcript_file_name = os.path.splitext(uploaded_file_name)[0] + '.txt'
    transcript_file_dir = os.path.join(tmp_document_dir, os.path.splitext(uploaded_file_name)[0])
    if not os.path.exists(transcript_file_dir):
        os.makedirs(transcript_file_dir)
    tmp_document_save_path = os.path.join(transcript_file_dir, transcript_file_name)
    with open(tmp_document_save_path, 'w') as f:
        f.write(full_document)
    return tmp_document_save_path


def extract_text_from_url(url):
    """

    :param url:
    :return:
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    text = text.replace('\n', ' ')
    cleaned_text = ' '.join(text.split())
    return cleaned_text
