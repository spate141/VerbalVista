import os
import pandas as pd
from glob import glob
import streamlit as st
# from spacy_streamlit import visualize_ner
from utils.other_utils import generate_wordcloud
from utils.rag_utils.rag_util import get_available_documents


def render_document_explore_page(document_dir=None, indices_dir=None, nlp=None, ner_labels=None):
    """
    This function will allow user to explore plain text to better understand the data.
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
                    st.markdown(f"<h6>File: {doc['filename']}</h6>", unsafe_allow_html=True)
                    st.markdown(f"<p>{' '.join(doc['text'].split())}</p>", unsafe_allow_html=True)
            with st.expander("Word Clouds", expanded=False):
                for doc in data:
                    st.markdown(f"<h6>File: {doc['filename']}</h6>", unsafe_allow_html=True)
                    plt = generate_wordcloud(text=doc['text'], background_color='black', colormap='Pastel1')
                    st.pyplot(plt)

            # for index, doc in enumerate(data):
                # doc = nlp(' '.join(doc['text'].split()))
                # visualize_ner(doc, labels=ner_labels, show_table=False, key=f"doc_{index}")
