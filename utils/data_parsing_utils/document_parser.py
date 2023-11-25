import re
import os
import tempfile
import docx2txt
from io import BytesIO
from pypdf import PdfReader
from langchain.document_loaders import UnstructuredEmailLoader

from utils import log_debug


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

