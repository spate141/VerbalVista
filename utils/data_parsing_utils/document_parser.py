import re
import email
import docx2txt
from io import BytesIO
from pypdf import PdfReader

from utils import log_error


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
    try:
        # Read the email file
        email_content = file.read().decode("utf-8")

        # Parse the email content
        msg = email.message_from_string(email_content)

        # Extract subject
        subject = msg['subject']

        # Extract email body
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                # Skip any text not in plain text or html
                if "text/plain" in content_type or "text/html" in content_type:
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            # Not a multipart message, simply get the payload
            body = msg.get_payload(decode=True).decode()

        text = subject + '\n\n' + body

    except Exception as e:
        log_error(f"An error occurred: {e}")
        text = None

    return text

