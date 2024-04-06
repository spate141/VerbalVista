import os
import time
import pandas as pd
import streamlit as st
from typing import Optional, List
from utils.rag_utils.rag_util import get_available_documents, index_data, delete_directory


def render_manage_index_page(
    document_dir: Optional[str] = None, indices_dir: Optional[str] = None, embedding_models: Optional[List[str]] = None
) -> None:
    """
    Renders a page in a Streamlit application that allows users to manage (create or delete) document indices.
    Users can select to create an index for documents using a specified embedding model and chunk size,
    or delete an existing index. The function also displays a list of available documents and their indices.

    :param document_dir: Optional; the directory where documents are stored. Defaults to None.
    :param indices_dir: Optional; the directory where document indices are stored. Defaults to None.
    :param embedding_models: Optional; a list of embedding model names available for creating document indices.
                             Defaults to None.
    :return: None
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
            embedding_model = st.selectbox(
                "embedding_model:", options=embedding_models, index=1, label_visibility="collapsed"
            )
        with cols[1]:
            st.markdown("<h6>Chunk Size:</h6>", unsafe_allow_html=True)
            chunk_size = st.number_input("chunk_size:", value=600, label_visibility="collapsed")
        st.markdown("</br>", unsafe_allow_html=True)

    elif mode == "Delete":
        mode_label = 'Deleting'
        pass

    st.markdown("<h6>Available Documents:</h6>", unsafe_allow_html=True)
    documents_df = get_available_documents(
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
                ]['Directory Name'].to_list()
                for doc_dir_to_index in document_dirs:
                    file_name = os.path.basename(doc_dir_to_index)
                    if mode == 'Create':
                        index_data(
                            document_directory=doc_dir_to_index,
                            index_directory=os.path.join(indices_dir, file_name),
                            chunk_size=chunk_size,
                            embedding_model=embedding_model
                        )
                        st.success(f"Document index {file_name} saved! Refreshing page now.")
                    elif mode == 'Delete':
                        _ = delete_directory(
                            selected_directory=os.path.join(indices_dir, file_name)
                        )
                        _ = delete_directory(
                            selected_directory=os.path.join(document_dir, file_name)
                        )
                        st.error(f"Document index {file_name} deleted! Refreshing page now.")

        time.sleep(2)
        st.rerun()
