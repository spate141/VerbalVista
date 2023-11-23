import os
import time
import streamlit as st
from utils.logging_module import log_info, log_debug, log_error
from utils.document_parser import parse_docx, parse_pdf, parse_txt, parse_email, parse_url, write_data_to_file


def render_media_processing_page(document_dir=None, tmp_audio_dir=None, audio_model=None, reddit_util=None):
    """
    This function will extract plain text from variety of media including video, audio, pdf and lots more.
    """
    supported_formats = ['m4a', 'mp3', 'wav', 'webm', 'mp4', 'mpg', 'mpeg', 'docx', 'pdf', 'txt', 'eml']
    st.header("Process Media", divider='violet')
    with st.form('docs_processing'):

        st.markdown(f"<h6>Data description: (optional)</h6>", unsafe_allow_html=True)
        document_desc = st.text_input("desc", placeholder="Enter data description", label_visibility="collapsed")

        st.markdown(f"<h6>Process file:</h6>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload file:", type=supported_formats, accept_multiple_files=False,
            label_visibility="collapsed"
        )

        st.markdown("<h6>Extract text from URL:</h6>", unsafe_allow_html=True)
        url = st.text_input("Enter URL:", placeholder='https://YOUR_URL', label_visibility="collapsed")
        url = None if len(url) == 0 else url

        st.markdown("<h6>Copy/Paste text:</h6>", unsafe_allow_html=True)
        text = st.text_area("Paste text:", placeholder='YOUR TEXT', label_visibility="collapsed")
        text = None if len(text) == 0 else text

        submitted = st.form_submit_button("Process", type="primary")
        if submitted:
            full_document = ''
            msg = st.toast('Processing data...')
            if uploaded_file is not None:
                log_debug('Processing uploaded file!')
                if uploaded_file.name.endswith(('.m4a', '.mp3', '.wav', '.webm', '.mp4', '.mpga', '.mpeg')):
                    msg.toast(f'Processing audio/video data...')
                    with st.spinner('Processing audio. Please wait.'):
                        process_audio_bar = st.progress(0, text="Processing...")
                        # Save the uploaded file to the specified directory
                        tmp_audio_save_path = os.path.join(tmp_audio_dir, uploaded_file.name)
                        log_debug(f"tmp_save_path: {tmp_audio_save_path}")
                        with open(tmp_audio_save_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        # Generate audio chunks
                        audio_chunks_files, file_size_mb, file_duration_in_ms = audio_model.generate_audio_chunks(
                            audio_filepath=tmp_audio_save_path, max_audio_size=25, tmp_dir=tmp_audio_dir,
                            process_bar=process_audio_bar
                        )
                        st.markdown(f"""
                        #### Audio Meta:
                        - Audio file size: {round(file_size_mb, 2)} MB
                        - Audio file duration: {audio_model.convert_milliseconds(int(file_duration_in_ms))}
                        """)

                    # Get transcript
                    start = time.time()
                    full_document = []
                    with st.spinner('Transcribing audio. Please wait.'):
                        transcribe_audio_bar = st.progress(0, text="Transcribing...")
                        total_chunks = len(audio_chunks_files)
                        pct_cmp = [i / total_chunks for i in range(1, total_chunks + 1)]
                        for index, i in enumerate(audio_chunks_files):
                            transcript = audio_model.transcribe_audio(i)
                            full_document.append(transcript)
                            transcribe_audio_bar.progress(
                                pct_cmp[index - 1], f'Audio transcribed: {round(time.time() - start, 2)} sec'
                            )
                        full_document = ' '.join(full_document)

                    # Remove tmp audio files
                    log_debug(f"Removing tmp audio files")
                    for file_name in os.listdir(tmp_audio_dir):
                        file_path = os.path.join(tmp_audio_dir, file_name)
                        if os.path.isfile(file_path):
                            os.remove(file_path)

                elif uploaded_file.name.endswith(".pdf"):
                    msg.toast(f'Processing PDF data...')
                    with st.spinner('Processing pdf file. Please wait.'):
                        full_document = parse_pdf(uploaded_file)

                elif uploaded_file.name.endswith(".docx"):
                    msg.toast(f'Processing DOCX data...')
                    with st.spinner('Processing word file. Please wait.'):
                        full_document = parse_docx(uploaded_file)

                elif uploaded_file.name.endswith(".txt"):
                    msg.toast(f'Processing TXT data...')
                    with st.spinner('Processing text file. Please wait.'):
                        full_document = parse_txt(uploaded_file)

                elif uploaded_file.name.endswith(".eml"):
                    msg.toast(f'Processing EMAIL data...')
                    with st.spinner('Processing email file. Please wait.'):
                        full_document = parse_email(uploaded_file)

                uploaded_file_name = uploaded_file.name.replace('.', '_').replace(' ', '_')

            elif url is not None:
                if "reddit.com" in url:
                    msg.toast(f'Processing Reddit post...')
                    log_debug('Processing Reddit post!')
                    full_document = reddit_util.fetch_comments_from_url(url)
                    full_document = ' '.join(full_document)
                else:
                    log_debug('Processing URL!')
                    full_document = parse_url(url, msg)
                uploaded_file_name = url[8:].replace("/", "-").replace('.', '-')

            elif text is not None:
                msg.toast(f'Processing TEXT data...')
                log_debug('Processing Text!')
                full_document = text
                uploaded_file_name = text[:20].replace("/", "-").replace('.', '-')

            else:
                st.error("You have to either upload a file, URL or enter some text!")
                return

            if len(full_document) == 0:
                st.error("No content available! Try something else.")
                return

            else:
                # Write document to a file
                st.markdown("#### Document snippet:")
                st.caption(full_document[:110] + '...')
                tmp_document_save_path = write_data_to_file(
                    uploaded_file_name=uploaded_file_name,
                    document_dir=document_dir,
                    full_document=full_document,
                    document_desc=document_desc
                )
                st.success(f"Document saved: {tmp_document_save_path}")
