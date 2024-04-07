import os
import shutil
import base64
import asyncio
import streamlit as st
from typing import Optional
from streamlit_theme import st_theme
from utils import log_info, log_debug, log_error
from utils.data_parsing_utils.document_parser import process_audio_files, process_document_files
from utils.data_parsing_utils import write_data_to_file
from utils.data_parsing_utils.url_parser import process_url, url_to_filename
from utils.data_parsing_utils.code_parser import CodeParser


def get_local_file_data(filepath: str) -> str:
    """
    Reads a local file and returns its contents encoded as a base64 URL.

    :param filepath: The path to the file to be read.
    :return: A string containing the base64 URL of the file's contents.
    """
    with open(filepath, "rb") as file_:
        contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    return data_url


def render_media_processing_page(
    document_dir: Optional[str] = None, tmp_dir: Optional[str] = None,
    openai_wisper_util: Optional[object] = None, reddit_util: Optional[object] = None
) -> None:
    """
    Renders a Streamlit page for processing various types of media including audio, video, PDFs, text, and URLs
    to extract plain text. Supports processing through file upload, URL input, GitHub repository cloning,
    or direct text input.

    :param document_dir: The directory where processed documents should be stored.
    :param tmp_dir: A temporary directory used for processing files.
    :param openai_wisper_util: Utility object for processing audio files using OpenAI's Whisper model.
    :param reddit_util: Utility object for fetching comments from Reddit URLs.
    :return: None
    """
    supported_formats = ['m4a', 'mp3', 'wav', 'webm', 'mp4', 'mpg', 'mpeg', 'docx', 'pdf', 'txt', 'eml']
    st.header("Process Media", divider='violet')
    col1, col2 = st.columns(2)
    extracted_text = ''
    document_desc = ''
    tmp_document_save_path = ''
    button_pressed = False
    with col1:
        with st.form('docs_processing'):
            st.markdown(f"<h6>Data description: (optional)</h6>", unsafe_allow_html=True)
            document_desc = st.text_input("desc", placeholder="Enter data description", label_visibility="collapsed")

            st.markdown(f"<h6>Process files:</h6>", unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Upload file:", type=supported_formats, accept_multiple_files=True,
                label_visibility="collapsed"
            )
            single_file_flag = st.checkbox("SAVE AS SINGLE FILE", value=False)
            st.markdown("<h6>Extract text from URL:</h6>", unsafe_allow_html=True)
            url = st.text_input("Enter URL:", placeholder='https://YOUR_URL', label_visibility="collapsed")
            url = None if len(url) == 0 else url

            st.markdown("<h6>Clone GitHub Repo (.git):</h6>", unsafe_allow_html=True)
            github_url = st.text_input("Enter .git URL:", placeholder='https://github.com/ORG/REPO.git', label_visibility="collapsed")
            github_url = None if len(github_url) == 0 else github_url

            st.markdown("<h6>Copy/Paste text:</h6>", unsafe_allow_html=True)
            text = st.text_area("Paste text:", placeholder='YOUR TEXT', label_visibility="collapsed")
            text = None if len(text) == 0 else text

            submitted = st.form_submit_button("Process", type="primary")
            if submitted:
                button_pressed = True
                msg = st.toast('Processing data...')
                full_documents = []
                processed_flag = False
                repo_name = None

                if uploaded_files:
                    all_files = [{
                        'name': file.name, 'type': file.type, 'size': file.size, 'file': file} for file in uploaded_files
                    ]
                    log_debug(f'Processing {len(all_files)} files.')

                    for file_meta in all_files:
                        file_name = file_meta['name']
                        extracted_text = ""

                        if file_name.endswith(('.m4a', '.mp3', '.wav', '.webm', '.mp4', '.mpga', '.mpeg')):
                            msg.toast(f'Processing audio/video data...')
                            with st.spinner('Transcribing audio. Please wait.'):
                                extracted_text = process_audio_files(
                                    tmp_dir=tmp_dir, file_meta=file_meta, openai_wisper_util=openai_wisper_util
                                )

                        elif file_name.endswith(('.pdf', '.docx', '.txt', '.eml')):
                            msg.toast(f'Processing file data...')
                            with st.spinner('Processing data file. Please wait.'):
                                extracted_text = process_document_files(file_meta=file_meta)

                        full_documents.append({
                            "file_name": file_name,
                            "extracted_text": extracted_text,
                            "doc_description": document_desc
                        })
                    processed_flag = True

                if url is not None:
                    if "reddit.com" in url:
                        msg.toast(f'Processing Reddit post...')
                        log_debug('Processing Reddit post!')
                        extracted_text = reddit_util.fetch_comments_from_url(url)
                        extracted_text = ' '.join(extracted_text)
                    else:
                        log_debug('Processing URL!')
                        extracted_text = asyncio.run(process_url(url, msg))

                    full_documents.append({
                        "file_name": url_to_filename(url),
                        "extracted_text": extracted_text,
                        "doc_description": document_desc
                    })
                    processed_flag = True

                if github_url is not None:
                    msg.toast(f'Processing GitHub repo...')
                    cp = CodeParser()
                    code_directory, repo_description, repo_name = cp.clone_or_update_repo(
                        tmp_dir=tmp_dir, git_repo_url=github_url
                    )
                    all_files = cp.find_code_files(
                        code_dir=code_directory, allowed_file_extensions=['.py', '.md']
                    )
                    full_documents = cp.extract_code(
                        code_filepaths=all_files, repo_description=repo_description
                    )
                    extracted_text = f"{github_url}\n\n{repo_description}..."
                    if os.path.exists(code_directory) and os.path.isdir(code_directory):
                        try:
                            shutil.rmtree(code_directory)
                        except Exception as e:
                            raise f"Error: Failed to remove {code_directory}. Reason: {e}"
                    processed_flag = True

                if text is not None:
                    msg.toast(f'Processing TEXT data...')
                    log_debug('Processing Text!')
                    extracted_text = text[:20].replace("/", "-").replace('.', '-')
                    full_documents.append({
                        "file_name": extracted_text,
                        "extracted_text": text,
                        "doc_description": document_desc
                    })
                    processed_flag = True

                if not processed_flag:
                    st.error("You have to either upload a file, URL or enter some text!")
                    return

                if len(full_documents) == 0:
                    st.error("No content available! Try something else.")
                    return

                else:
                    # Write document to a file
                    # st.markdown("#### Document snippet:")
                    tmp_document_save_path = write_data_to_file(
                        document_dir=document_dir,
                        full_documents=full_documents,
                        single_file_flag=single_file_flag,
                        save_dir_name=repo_name if repo_name else None
                    )
                    # st.success(f"Document saved: {tmp_document_save_path}")

    with col2:
        with st.container(border=True, height=590):
            st.markdown('<h6>Extracted text preview:<h6>', unsafe_allow_html=True)
            if button_pressed:
                if tmp_document_save_path:
                    st.success(f"Document saved: {tmp_document_save_path}")
                if document_desc:
                    st.info(f'{document_desc}')
                if extracted_text:
                    st.markdown(f'{extracted_text[:1230]}...')
            else:
                theme = st_theme()
                if theme and theme['backgroundColor'] == '#ffffff':
                    data_url = get_local_file_data('docs/loading_white.gif')
                else:
                    data_url = get_local_file_data('docs/loading_black.gif')
                cols = st.columns([0.7, 10, 0.5])
                with cols[1]:
                    st.markdown(
                        f'<a href="https://github.com/spate141/VerbalVista"><img src="data:image/gif;base64,{data_url}" alt="loading" class="center"></a>',
                        unsafe_allow_html=True,
                    )
