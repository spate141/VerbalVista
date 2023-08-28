import os
import spacy
import streamlit as st
from app_pages import *
from utils.ask_util import AskUtil
from utils.stocks_util import StockUtil
from utils.indexing_util import IndexUtil
from utils.summary_util import SummaryUtil
from utils.google_serper_util import GoogleSerperUtil
from utils.audio_transcribe import WhisperAudioTranscribe
from utils.logging_module import log_info, log_debug, log_error


class VerbalVista:

    def __init__(
            self, document_dir: str = None, tmp_audio_dir: str = None, indices_dir: str = None,
            chat_history_dir: str = None, search_history_dir: str = None, stock_data_dir: str = None
    ):

        # Initialize all necessary classes
        self.whisper = WhisperAudioTranscribe()
        self.indexing_util = IndexUtil()
        self.ask_util = AskUtil()
        self.summary_util = SummaryUtil()
        self.google_serper_util = GoogleSerperUtil()
        self.stock_util = StockUtil()

        # Create relevant directories
        for directory_path in [
            document_dir, indices_dir, tmp_audio_dir, chat_history_dir,
            search_history_dir, stock_data_dir
        ]:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                log_debug(f"Directory '{directory_path}' created successfully.")

        # Initialize common variables, models
        self.document_dir = document_dir
        self.indices_dir = indices_dir
        self.tmp_audio_dir = tmp_audio_dir
        self.chat_history_dir = chat_history_dir
        self.search_history_dir = search_history_dir
        self.stock_data_dir = stock_data_dir
        self.nlp = spacy.load("en_core_web_sm")
        self.ner_labels = self.nlp.get_pipe("ner").labels

    def render_media_processing_page(self):

        render_media_processing_page(
            document_dir=self.document_dir, tmp_audio_dir=self.tmp_audio_dir, audio_model=self.whisper
        )

    def render_manage_index_page(self):

        render_manage_index_page(
            document_dir=self.document_dir, indices_dir=self.indices_dir, indexing_util=self.indexing_util
        )

    def render_document_explore_page(self):

        render_document_explore_page(
            document_dir=self.document_dir, indices_dir=self.indices_dir,
            indexing_util=self.indexing_util, nlp=self.nlp, ner_labels=self.ner_labels
        )

    def render_qa_page(self, temperature=None, max_tokens=None, model_name=None, chain_type=None):

        render_qa_page(
            temperature=temperature, max_tokens=max_tokens, model_name=model_name, chain_type=chain_type,
            ask_util=self.ask_util, indexing_util=self.indexing_util, summary_util=self.summary_util,
            indices_dir=self.indices_dir, document_dir=self.document_dir, chat_history_dir=self.chat_history_dir
        )

    def render_tell_me_about_page(self):
        """
        Tell me more about page.
        """
        render_tell_me_about_page(
            google_serper_util=self.google_serper_util, summary_util=self.summary_util,
            search_history_dir=self.search_history_dir
        )

    def render_stocks_comparison_page(self):
        """
        Stocks comparison page.
        """
        render_stocks_comparison_page(stock_util=self.stock_util, stock_data_dir=self.stock_data_dir)


def main():
    APP_NAME = "VerbalVista"
    APP_VERSION = '0.0.7'
    APP_PAGES = [
        "Media Processing",
        "Explore Document",
        "Manage Index",
        "Q & A",
        "Tell Me About",
        "Stocks Comparison"
    ]
    # Render sidebar
    openai_api_key, selected_page = render_sidebar(
        app_name=APP_NAME, app_version=APP_VERSION, app_pages=APP_PAGES
    )

    # Project local cache directories
    document_dir = 'data/documents/'
    tmp_audio_dir = 'data/tmp_audio_dir/'
    indices_dir = 'data/indices/'
    chat_history_dir = 'data/chat_history/'
    search_history_dir = 'data/search_history/'
    stock_data_dir = 'data/stock_data_dir/'

    if not os.environ.get("OPENAI_API_KEY", None) and not openai_api_key:
        # if both env variable and explicit key is not set
        st.error("OpenAI API key not found!")
        log_error("No OpenAI key found!")
    elif not os.environ.get("OPENAI_API_KEY", None) and openai_api_key:
        # if env variable is not set but user provide explicit key
        os.environ['OPENAI_API_KEY'] = openai_api_key
        log_info(f"No OpenAI key found in ENV, User provided key.")
    elif os.environ.get("OPENAI_API_KEY", None) and openai_api_key:
        # if both env variable and explicit keys are provided
        os.environ['OPENAI_API_KEY'] = openai_api_key
        log_info(f"OpenAI key found in ENV & User provided key.")

    vv = VerbalVista(
        document_dir=document_dir, indices_dir=indices_dir,
        tmp_audio_dir=tmp_audio_dir, chat_history_dir=chat_history_dir,
        search_history_dir=search_history_dir, stock_data_dir=stock_data_dir
    )
    if selected_page == "Media Processing":
        vv.render_media_processing_page()
    elif selected_page == "Manage Index":
        vv.render_manage_index_page()
    elif selected_page == "Q & A":
        with st.sidebar:
            temperature = st.number_input("Temperature", value=0.5, min_value=0.0, max_value=1.0)
            max_tokens = st.number_input("Max Tokens", value=512, min_value=0, max_value=4000)
            model_name = st.selectbox("Model Name", ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"], index=0)
            summ_chain_type = st.selectbox("Chain type", index=1, options=["stuff", "map_reduce", "refine"])
        vv.render_qa_page(temperature=temperature, max_tokens=max_tokens, model_name=model_name, chain_type=summ_chain_type)
    elif selected_page == "Explore Document":
        vv.render_document_explore_page()
    elif selected_page == 'Tell Me About':
        vv.render_tell_me_about_page()
    elif selected_page == 'Stocks Comparison':
        vv.render_stocks_comparison_page()


if __name__ == '__main__':
    main()
