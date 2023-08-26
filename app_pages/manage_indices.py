import os
import time
import pandas as pd
import streamlit as st
from utils.logging_module import log_info, log_debug, log_error


def render_manage_index_page(document_dir=None, indices_dir=None, indexing_util=None):
    """
    This function will allow user to convert plain text into vector index or remove already created index.
    """
    st.header("Manage Index", divider='green')
    st.markdown("<h6>Select Mode:</h6>", unsafe_allow_html=True)
    mode = st.selectbox("mode", ["Create", "Delete"], index=0, label_visibility="collapsed")
    mode_label = None

    if mode == "Create":
        mode_label = 'Creating'
        cols = st.columns(2)
        with cols[0]:
            st.markdown("<h6>Select Embedding Model:</h6>", unsafe_allow_html=True)
            embedding_model = st.selectbox("embedding_model:", options=[
                "text-embedding-ada-002"
            ], index=0, label_visibility="collapsed")
        with cols[1]:
            st.markdown("<h6>Chunk Size:</h6>", unsafe_allow_html=True)
            chunk_size = st.number_input("chunk_size:", value=600, label_visibility="collapsed")
        st.markdown("</br>", unsafe_allow_html=True)

    elif mode == "Delete":
        mode_label = 'Deleting'
        pass

    st.markdown("<h6>Available Documents:</h6>", unsafe_allow_html=True)
    documents_df = indexing_util.get_available_documents(
        document_dir=document_dir, indices_dir=indices_dir
    )
    documents_df['Creation Date'] = pd.to_datetime(documents_df['Creation Date'])
    documents_df = documents_df.sort_values(by='Creation Date', ascending=False)
    selected_documents_df = st.data_editor(documents_df, hide_index=True, use_container_width=True, height=300)

    submit = st.button("Submit", type="primary")
    if submit:
        _, c, _ = st.columns([2, 5, 2])
        with c:
            with st.spinner(f'{mode_label} document. Please wait.'):
                document_dirs = selected_documents_df[
                    selected_documents_df['Select Index']
                ]['Document Name'].to_list()
                for doc_dir_to_index in document_dirs:
                    file_name = os.path.splitext(os.path.basename(doc_dir_to_index))[0]
                    if mode == 'Create':
                        indexing_meta = indexing_util.index_document(
                            document_directory=doc_dir_to_index,
                            index_directory=os.path.join(indices_dir, file_name),
                            chunk_size=chunk_size,
                            embedding_model=embedding_model
                        )
                        st.success(f"Document index {file_name} saved! Refreshing page now.")
                        st.info(indexing_meta)
                    elif mode == 'Delete':
                        indexing_util.delete_document(
                            selected_directory=os.path.join(indices_dir, file_name)
                        )
                        indexing_util.delete_document(
                            selected_directory=os.path.join(document_dir, file_name)
                        )
                        st.error(f"Document index {file_name} deleted! Refreshing page now.")

        time.sleep(2)
        st.experimental_rerun()
