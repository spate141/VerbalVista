import asyncio
import streamlit as st
from utils import log_info, log_debug, log_error
from utils.data_parsing_utils.document_parser import process_audio_files, process_document_files
from utils.data_parsing_utils import write_data_to_file
from utils.data_parsing_utils.url_parser import process_url, url_to_filename


def render_media_processing_page(document_dir=None, tmp_audio_dir=None, openai_wisper_util=None, reddit_util=None):
    """
    This function will extract plain text from variety of media including video, audio, pdf and lots more.
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

            st.markdown("<h6>Copy/Paste text:</h6>", unsafe_allow_html=True)
            text = st.text_area("Paste text:", placeholder='YOUR TEXT', label_visibility="collapsed")
            text = None if len(text) == 0 else text

            submitted = st.form_submit_button("Process", type="primary")
            if submitted:
                button_pressed = True
                msg = st.toast('Processing data...')
                full_documents = []
                processed_flag = False
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
                                    tmp_audio_dir=tmp_audio_dir, file_meta=file_meta, openai_wisper_util=openai_wisper_util
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
                st.markdown("![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExY25wdWh3YXFvZXIzcWRtMnIwcXl3dTk1eGF6bmczd3hiYTY2NzcxNCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/No3Q2COl8SEnu/giphy.gif)")


