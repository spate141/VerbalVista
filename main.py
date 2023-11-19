import os
import spacy
import streamlit as st
from app_pages import *
from dotenv import load_dotenv
from utils.ask_util import AskUtil
from utils.stocks_util import StockUtil
from utils.indexing_util import IndexUtil
from utils.summary_util import SummaryUtil
from utils.text_to_speech import TextToSpeech
from utils.google_serper_util import GoogleSerperUtil
from utils.reddit_util import SubmissionCommentsFetcher
from utils.image_generation_util import ImageGeneration
from utils.audio_transcribe import WhisperAudioTranscribe
from utils.logging_module import log_info, log_debug, log_error


class VerbalVista:

    def __init__(
            self, document_dir: str = None, tmp_audio_dir: str = None, indices_dir: str = None,
            chat_history_dir: str = None, search_history_dir: str = None, stock_data_dir: str = None,
            generated_images_dir: str = None
    ):

        # Initialize all necessary classes
        self.whisper = WhisperAudioTranscribe()
        self.tx2sp_util = TextToSpeech()
        self.indexing_util = IndexUtil()
        self.ask_util = AskUtil()
        self.summary_util = SummaryUtil()
        self.google_serper_util = GoogleSerperUtil()
        self.stock_util = StockUtil()
        self.image_generation_util = ImageGeneration()
        self.reddit_util = SubmissionCommentsFetcher(
            os.getenv('REDDIT_CLIENT_ID'),
            os.getenv('REDDIT_CLIENT_SECRET'),
            os.getenv('REDDIT_USER_AGENT')
        )

        # Create relevant directories
        for directory_path in [
            document_dir, indices_dir, tmp_audio_dir, chat_history_dir,
            search_history_dir, stock_data_dir, generated_images_dir
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
        self.generated_images_dir = generated_images_dir
        self.nlp = spacy.load("en_core_web_sm")
        self.ner_labels = self.nlp.get_pipe("ner").labels

    def render_media_processing_page(self):
        """
        Media input and processing page.
        """
        render_media_processing_page(
            document_dir=self.document_dir, tmp_audio_dir=self.tmp_audio_dir, audio_model=self.whisper,
            reddit_util=self.reddit_util
        )

    def render_manage_index_page(self):
        """
        Create/Manage/Delete document index page.
        """
        render_manage_index_page(
            document_dir=self.document_dir, indices_dir=self.indices_dir, indexing_util=self.indexing_util
        )

    def render_document_explore_page(self):
        """
        Document explore page.
        """
        render_document_explore_page(
            document_dir=self.document_dir, indices_dir=self.indices_dir,
            indexing_util=self.indexing_util, nlp=self.nlp, ner_labels=self.ner_labels
        )

    def render_qa_page(self, temperature=None, max_tokens=None, model_name=None, chain_type=None, enable_tts=False, tts_voice=None):
        """
        Question answer page.
        """
        render_qa_page(
            temperature=temperature, max_tokens=max_tokens, model_name=model_name, chain_type=chain_type,
            ask_util=self.ask_util, indexing_util=self.indexing_util, summary_util=self.summary_util, tx2sp_util=self.tx2sp_util,
            indices_dir=self.indices_dir, document_dir=self.document_dir, chat_history_dir=self.chat_history_dir,
            enable_tts=enable_tts, tts_voice=tts_voice
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

    def render_image_generation_page(self):
        """
        Image generation page.
        """
        render_image_generation_page(
            generated_images_dir=self.generated_images_dir, image_generation_util=self.image_generation_util
        )


def main():
    APP_NAME = "VerbalVista"
    APP_VERSION = "1.1"
    APP_PAGES = [
        "Media Processing", "Explore Document", "Manage Index", "Q & A", "Tell Me About", "Stocks Comparison",
        "Image Generation"
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
    generated_images_dir = 'data/generated_images/'

    # Load env variables
    load_dotenv()

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
        search_history_dir=search_history_dir, stock_data_dir=stock_data_dir,
        generated_images_dir=generated_images_dir
    )
    if selected_page == "Media Processing":
        vv.render_media_processing_page()
    elif selected_page == "Manage Index":
        vv.render_manage_index_page()
    elif selected_page == "Q & A":
        with st.sidebar:
            temperature = st.number_input("Temperature", value=0.5, min_value=0.0, max_value=1.0)
            max_tokens = st.number_input("Max Tokens", value=512, min_value=0, max_value=4000)
            model_name = st.selectbox("Model Name", ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k", "gpt-4-1106-preview"], index=2)
            summ_chain_type = st.selectbox("Chain type", index=1, options=["stuff", "map_reduce", "refine"])
            enable_tts = st.checkbox("Enable text-to-speech", value=False)
            tts_voice = "echo"
            if enable_tts:
                tts_voice = st.selectbox("Select Voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], index=1)
        vv.render_qa_page(
            temperature=temperature, max_tokens=max_tokens, model_name=model_name,
            chain_type=summ_chain_type, enable_tts=enable_tts, tts_voice=tts_voice
        )
    elif selected_page == "Explore Document":
        vv.render_document_explore_page()
    elif selected_page == 'Tell Me About':
        vv.render_tell_me_about_page()
    elif selected_page == 'Stocks Comparison':
        vv.render_stocks_comparison_page()
    elif selected_page == 'Image Generation':
        vv.render_image_generation_page()


if __name__ == '__main__':
    main()
