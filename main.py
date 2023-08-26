import os
import spacy
from PIL import Image
import streamlit as st
from app_pages import *
from utils.ask_util import AskUtil
from utils.indexing_util import IndexUtil
from utils.summary_util import SummaryUtil
from utils.google_serper_util import GoogleSerperUtil
from utils.audio_transcribe import WhisperAudioTranscribe
from utils.logging_module import log_info, log_debug, log_error


class VerbalVista:

    def __init__(self, document_dir: str = None, tmp_audio_dir: str = None, indices_dir: str = None, chat_history_dir: str = None, search_history_dir: str = None):

        # Initialize all necessary classes
        self.whisper = WhisperAudioTranscribe()
        self.indexing_util = IndexUtil()
        self.ask_util = AskUtil()
        self.summary_util = SummaryUtil()
        self.google_serper_util = GoogleSerperUtil()

        # Create relevant directories
        for directory_path in [document_dir, indices_dir, tmp_audio_dir, chat_history_dir, search_history_dir]:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                log_debug(f"Directory '{directory_path}' created successfully.")

        # Initialize common variables, models
        self.document_dir = document_dir
        self.indices_dir = indices_dir
        self.tmp_audio_dir = tmp_audio_dir
        self.chat_history_dir = chat_history_dir
        self.search_history_dir = search_history_dir
        self.nlp = spacy.load("en_core_web_sm")
        self.ner_labels = self.nlp.get_pipe("ner").labels

    def render_media_processing_page(self):

        render_media_processing_page(
            document_dir=self.document_dir,
            tmp_audio_dir=self.tmp_audio_dir,
            audio_model=self.whisper
        )

    def render_manage_index_page(self):

        render_manage_index_page(
            document_dir=self.document_dir,
            indices_dir=self.indices_dir,
            indexing_util=self.indexing_util
        )

    def render_document_explore_page(self):

        render_document_explore_page(
            document_dir=self.document_dir,
            indices_dir=self.indices_dir,
            indexing_util=self.indexing_util,
            nlp=self.nlp,
            ner_labels=self.ner_labels
        )

    def render_qa_page(self, temperature=None, max_tokens=None, model_name=None, chain_type=None):

        render_qa_page(
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=model_name,
            chain_type=chain_type,
            ask_util=self.ask_util,
            indexing_util=self.indexing_util,
            summary_util=self.summary_util,
            indices_dir=self.indices_dir,
            document_dir=self.document_dir,
            chat_history_dir=self.chat_history_dir
        )

    def render_tell_me_about_page(self):
        """
        Tell me more about page.
        """
        render_tell_me_about_page(
            google_serper_util=self.google_serper_util,
            summary_util=self.summary_util,
            search_history_dir=self.search_history_dir
        )

    def render_stocks_comparison_page(self):
        """
        Stocks comparison page.
        """
        render_stocks_comparison_page()


def main():
    page_icon = Image.open('docs/logo-white.png')
    app_version = "0.0.7"
    st.set_page_config(
        page_title="VerbalVista",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/spate141/VerbalVista',
            'Report a bug': "https://github.com/spate141/VerbalVista",
            'About': "### Welcome to VerbalVista!\nBuilt by Snehal Patel."
        }
    )
    st.sidebar.markdown(
        f"""
        <center>
        <a href="https://github.com/spate141/VerbalVista"><img src="https://i.ibb.co/6FQPs5C/verbal-vista-blue-transparent.png" width="70%" height="70%"></a>
        </br>
        </br>
        <h5>Version: {app_version}</h5>
        </center>
        <i class="fa-regular fa-v fa-flip"></i>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("<center><h4><b>OpenAI API Key</b></h5></center>", unsafe_allow_html=True)
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key:", key="chatbot_api_key", type="password", label_visibility="collapsed"
    )
    st.sidebar.markdown("<center><h4><b>Select Function</b></h5></center>", unsafe_allow_html=True)
    page = st.sidebar.selectbox(
        "Select function:", [
            "Media Processing",
            "Explore Document",
            "Manage Index",
            "Q & A",
            "Tell Me About",
            "Stocks Comparison"
        ], label_visibility="collapsed"
    )

    document_dir = 'data/documents/'
    tmp_audio_dir = 'data/tmp_audio_dir/'
    indices_dir = 'data/indices/'
    chat_history_dir = 'data/chat_history/'
    search_history_dir = 'data/search_history/'

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
        search_history_dir=search_history_dir
    )
    if page == "Media Processing":
        vv.render_media_processing_page()
    elif page == "Manage Index":
        vv.render_manage_index_page()
    elif page == "Q & A":
        with st.sidebar:
            temperature = st.number_input("Temperature", value=0.5, min_value=0.0, max_value=1.0)
            max_tokens = st.number_input("Max Tokens", value=512, min_value=0, max_value=4000)
            model_name = st.selectbox("Model Name", ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"], index=0)
            summ_chain_type = st.selectbox("Chain type", index=1, options=["stuff", "map_reduce", "refine"])
        vv.render_qa_page(temperature=temperature, max_tokens=max_tokens, model_name=model_name, chain_type=summ_chain_type)
    elif page == "Explore Document":
        vv.render_document_explore_page()
    elif page == 'Tell Me About':
        vv.render_tell_me_about_page()
    elif page == 'Stocks Comparison':
        vv.render_stocks_comparison_page()


if __name__ == '__main__':
    main()
