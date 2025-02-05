import re
import os
import html
import pandas as pd
from glob import glob
import streamlit as st
from typing import Optional
from utils.other_utils import generate_wordcloud
from utils.rag_utils.rag_util import get_available_documents


def remove_non_english(text):
    """
    Remove non-English characters
    """
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Keeps only ASCII characters
    return html.escape(text)  # Escapes any remaining special HTML characters

def render_document_explore_page(document_dir: Optional[str] = None, indices_dir: Optional[str] = None) -> None:
    """
    Renders a page in a Streamlit application for exploring documents. This page allows users to select documents
    from a list (based on a directory of documents and an index directory), view the document text, and generate
    a word cloud for visual analysis.

    :param document_dir: Optional; the directory where documents are stored. Defaults to None.
    :param indices_dir: Optional; the directory where document indices are stored. Defaults to None.
    :return: None
    """
    st.header('Explore Document', divider='blue')
    with st.form('explore_document'):
        st.markdown("<h6>Select Document:</h6>", unsafe_allow_html=True)
        documents_df = get_available_documents(
            document_dir=document_dir, indices_dir=indices_dir
        )
        documents_df = documents_df.rename(columns={'Select Index': 'Select Document'})
        documents_df['Creation Date'] = pd.to_datetime(documents_df['Creation Date'])
        documents_df = documents_df.sort_values(by='Creation Date', ascending=False)
        selected_documents_df = st.data_editor(documents_df, hide_index=True, use_container_width=True, height=300)
        submitted = st.form_submit_button("Explore!", type="primary")
        if submitted:
            selected_docs_dir_paths = selected_documents_df[
                selected_documents_df['Select Document']
            ]['Directory Name'].to_list()
            data = []
            for selected_doc_dir_path in selected_docs_dir_paths:
                filenames = glob(f"{selected_doc_dir_path}/*.data.txt")
                for filename in filenames:
                    filename = os.path.basename(filename)
                    filepath = os.path.join(selected_doc_dir_path, filename)
                    with open(filepath, 'r') as f:
                        data.append({"filename": filename, "text": f.read()})

            with st.expander("Text", expanded=False):
                for doc in data:
                    clean_text = remove_non_english(doc['text'])
                    st.markdown(f"<h6>File: {doc['filename']}</h6>", unsafe_allow_html=True)
                    st.markdown(f"<p>{' '.join(clean_text.split())}</p>", unsafe_allow_html=True)
            with st.expander("Word Clouds", expanded=False):
                for doc in data:
                    st.markdown(f"<h6>File: {doc['filename']}</h6>", unsafe_allow_html=True)
                    plt = generate_wordcloud(text=doc['text'], background_color='black', colormap='Pastel1')
                    st.pyplot(plt)

