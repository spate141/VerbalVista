import os
import pandas as pd
import streamlit as st
from spacy_streamlit import visualize_ner
from utils.generate_wordcloud import generate_wordcloud
from utils.logging_module import log_info, log_debug, log_error


def render_document_explore_page(document_dir=None, indices_dir=None, rag_util=None, nlp=None, ner_labels=None):
    """
    This function will allow user to explore plain text to better understand the data.
    """
    st.header('Explore Document', divider='blue')
    with st.form('explore_document'):
        st.markdown("<h6>Select Document:</h6>", unsafe_allow_html=True)
        documents_df = rag_util.get_available_documents(
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
            ]['Document Name'].to_list()
            data = []
            for selected_doc_dir_path in selected_docs_dir_paths:
                filename = selected_doc_dir_path.split('/')[-1] + '.data.txt'
                filepath = os.path.join(selected_doc_dir_path, filename)
                with open(filepath, 'r') as f:
                    text = f.read()
                    data.append({"filename": filename, "text": text})

            with st.expander("Text", expanded=False):
                for doc in data:
                    st.markdown(f"<h6>File: {doc['filename']}</h6>", unsafe_allow_html=True)
                    st.markdown(f"<p>{' '.join(doc['text'].split())}</p>", unsafe_allow_html=True)
            with st.expander("Word Clouds", expanded=False):
                for doc in data:
                    st.markdown(f"<h6>File: {doc['filename']}</h6>", unsafe_allow_html=True)
                    plt = generate_wordcloud(text=doc['text'], background_color='black', colormap='Pastel1')
                    st.pyplot(plt)

            for index, doc in enumerate(data):
                doc = nlp(' '.join(doc['text'].split()))
                visualize_ner(doc, labels=ner_labels, show_table=False, key=f"doc_{index}")
