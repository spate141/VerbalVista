import re
import os
import email
import docx2txt
from io import BytesIO
from pypdf import PdfReader
from typing import Dict, Any, Optional
from utils import log_error, log_debug


def parse_docx(file: BytesIO) -> str:
    """
    Parses DOCX (Word) files and returns their content as a string.

    :param file: A BytesIO object containing the DOCX file.
    :return: Text content of the DOCX file.
    """
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def parse_pdf(file: BytesIO) -> str:
    """
    Parses PDF files and returns their content as a string.

    :param file: A BytesIO object containing the PDF file.
    :return: Text content of the PDF file.
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
    Parses TXT files and returns their content as a string.

    :param file: A BytesIO object containing the TXT file.
    :return: Text content of the TXT file.
    """
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def parse_email(file: BytesIO) -> Optional[str]:
    """
    Parses EML (Email) files and returns their content as a string.

    :param file: A BytesIO object containing the EML file.
    :return: Text content of the EML file or None if an error occurs.
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


def process_document_files(file_meta: Dict[str, Any]) -> str:
    """
    Processes document files based on their extension and returns the extracted text.

    :param file_meta: Metadata about the file including its name and the BytesIO object (`file`).
    :return: Extracted text from the document.
    """
    file_name = file_meta['name']
    file = file_meta['file']
    extracted_text = ""
    if file_name.endswith(".pdf"):
        log_debug('Processing pdf file. Please wait.')
        extracted_text = parse_pdf(file)

    elif file_name.endswith(".docx"):
        log_debug('Processing word file. Please wait.')
        extracted_text = parse_docx(file)

    elif file_name.endswith(".txt"):
        log_debug('Processing text file. Please wait.')
        extracted_text = parse_txt(file)

    elif file_name.endswith(".eml"):
        log_debug('Processing email file. Please wait.')
        extracted_text = parse_email(file)
    return extracted_text


def process_audio_files(tmp_dir: str, file_meta: Dict[str, Any], openai_wisper_util: Any) -> str:
    """
    Processes audio files, converting speech to text using a specified utility (e.g., OpenAI's Whisper).

    :param tmp_dir: The directory to temporarily save audio files.
    :param file_meta: Metadata about the file including its name and the BytesIO object (`file`).
    :param openai_wisper_util: The utility to use for speech-to-text conversion.
    :return: Transcribed text from the audio file.
    """

    file_name = file_meta['name']
    file = file_meta['file']

    # Save the uploaded file to the specified directory
    tmp_audio_save_path = os.path.join(tmp_dir, file_name)
    log_debug(f"tmp_save_path: {tmp_audio_save_path}")
    with open(tmp_audio_save_path, "wb") as f:
        f.write(file.getvalue())

    # Generate audio chunks
    audio_chunks_files, file_size_mb, file_duration_in_ms = openai_wisper_util.generate_audio_chunks(
        audio_filepath=tmp_audio_save_path, max_audio_size=25, tmp_dir=tmp_dir
    )

    # Get transcript for all chunks
    all_transcripts = []
    for index, i in enumerate(audio_chunks_files):
        transcript = openai_wisper_util.transcribe_audio(i)
        all_transcripts.append(transcript)

    # Create a single transcript from different chunks of audio
    full_transcript = ' '.join(all_transcripts)

    # Remove tmp audio files
    log_debug(f"Removing tmp audio files")
    for file_name in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return full_transcript


def process_image_files(files):
    """
    Process each image files, do ComputerVision and extract an embedding for each image
    and return processed data with metadata
    """
    pass

