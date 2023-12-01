import os
import time
import streamlit as st
from utils import log_info, log_debug, log_error
from utils.data_parsing_utils.document_parser import parse_docx, parse_pdf, parse_txt, parse_email
from utils.data_parsing_utils import write_data_to_file
from utils.data_parsing_utils.url_parser import parse_url


def render_media_processing_page(document_dir=None, tmp_audio_dir=None, audio_model=None, reddit_util=None):
    """
    This function will extract plain text from variety of media including video, audio, pdf and lots more.
    """
    supported_formats = ['m4a', 'mp3', 'wav', 'webm', 'mp4', 'mpg', 'mpeg', 'docx', 'pdf', 'txt', 'eml']
    st.header("Process Media", divider='violet')
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
            msg = st.toast('Processing data...')
            full_documents = []
            if uploaded_files:
                all_files = [{'name': file.name, 'type': file.type, 'size': file.size, 'file': file} for file in uploaded_files]
                log_debug(f'Processing {len(all_files)}')

                for file_meta in all_files:
                    file_name = file_meta['name']
                    file = file_meta['file']
                    extracted_text = ""

                    if file_name.endswith(('.m4a', '.mp3', '.wav', '.webm', '.mp4', '.mpga', '.mpeg')):
                        msg.toast(f'Processing audio/video data...')
                        with st.spinner('Processing audio. Please wait.'):
                            process_audio_bar = st.progress(0, text="Processing...")
                            # Save the uploaded file to the specified directory
                            tmp_audio_save_path = os.path.join(tmp_audio_dir, file_name)
                            log_debug(f"tmp_save_path: {tmp_audio_save_path}")
                            with open(tmp_audio_save_path, "wb") as f:
                                f.write(file.getvalue())
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

                        # Get transcript for all chunks
                        start = time.time()
                        all_transcripts = []
                        with st.spinner('Transcribing audio. Please wait.'):
                            transcribe_audio_bar = st.progress(0, text="Transcribing...")
                            total_chunks = len(audio_chunks_files)
                            pct_cmp = [i / total_chunks for i in range(1, total_chunks + 1)]
                            for index, i in enumerate(audio_chunks_files):
                                transcript = audio_model.transcribe_audio(i)
                                all_transcripts.append(transcript)
                                transcribe_audio_bar.progress(
                                    pct_cmp[index - 1], f'Audio transcribed: {round(time.time() - start, 2)} sec'
                                )

                        # Remove tmp audio files
                        log_debug(f"Removing tmp audio files")
                        for file_name in os.listdir(tmp_audio_dir):
                            file_path = os.path.join(tmp_audio_dir, file_name)
                            if os.path.isfile(file_path):
                                os.remove(file_path)

                        # Create a single transcript from different chunks of audio
                        extracted_text = ' '.join(all_transcripts)

                    elif file_name.endswith(".pdf"):
                        msg.toast(f'Processing PDF data...')
                        with st.spinner('Processing pdf file. Please wait.'):
                            extracted_text = parse_pdf(file)

                    elif file_name.endswith(".docx"):
                        msg.toast(f'Processing DOCX data...')
                        with st.spinner('Processing word file. Please wait.'):
                            extracted_text = parse_docx(file)

                    elif file_name.endswith(".txt"):
                        msg.toast(f'Processing TXT data...')
                        with st.spinner('Processing text file. Please wait.'):
                            extracted_text = parse_txt(file)

                    elif file_name.endswith(".eml"):
                        msg.toast(f'Processing EMAIL data...')
                        with st.spinner('Processing email file. Please wait.'):
                            extracted_text = parse_email(file)

                    full_documents.append({
                        "file_name": file_name,
                        "extracted_text": extracted_text,
                        "doc_description": document_desc
                    })

            elif url is not None:
                if "reddit.com" in url:
                    msg.toast(f'Processing Reddit post...')
                    log_debug('Processing Reddit post!')
                    extracted_text = reddit_util.fetch_comments_from_url(url)
                    extracted_text = ' '.join(extracted_text)
                else:
                    log_debug('Processing URL!')
                    extracted_text = parse_url(url, msg)

                full_documents.append({
                    "file_name": url[8:].replace("/", "-").replace('.', '-'),
                    "extracted_text": extracted_text,
                    "doc_description": document_desc
                })

            elif text is not None:
                msg.toast(f'Processing TEXT data...')
                log_debug('Processing Text!')
                full_documents.append({
                    "file_name": text[:20].replace("/", "-").replace('.', '-'),
                    "extracted_text": text,
                    "doc_description": document_desc
                })
            else:
                st.error("You have to either upload a file, URL or enter some text!")
                return

            if len(full_documents) == 0:
                st.error("No content available! Try something else.")
                return

            else:
                # Write document to a file
                st.markdown("#### Document snippet:")
                # st.caption(full_document[:110] + '...')
                tmp_document_save_path = write_data_to_file(
                    document_dir=document_dir,
                    full_documents=full_documents,
                    single_file_flag=single_file_flag,
                )
                st.success(f"Document saved: {tmp_document_save_path}")
