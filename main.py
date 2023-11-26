import os
import spacy
import streamlit as st
from app_pages import *
from dotenv import load_dotenv; load_dotenv()

from utils import log_info, log_debug, log_error
from utils.data_parsing_utils import RedditSubmissionCommentsFetcher
from utils.openai_utils import OpenAIDalleUtil, OpenAIWisperUtil, OpenAIText2SpeechUtil


class VerbalVista:

    def __init__(
            self, document_dir: str = None, tmp_audio_dir: str = None, indices_dir: str = None,
            chat_history_dir: str = None, stock_data_dir: str = None, generated_images_dir: str = None
    ):

        # Initialize all necessary classes
        self.openai_wisper_util = OpenAIWisperUtil(api_key=os.getenv("OPENAI_API_KEY"))
        self.openai_t2s_util = OpenAIText2SpeechUtil(api_key=os.getenv("OPENAI_API_KEY"))
        self.openai_dalle_util = OpenAIDalleUtil(api_key=os.getenv("OPENAI_API_KEY"))
        self.reddit_util = RedditSubmissionCommentsFetcher(
            os.getenv('REDDIT_CLIENT_ID'),
            os.getenv('REDDIT_CLIENT_SECRET'),
            os.getenv('REDDIT_USER_AGENT')
        )

        # Create relevant directories
        for directory_path in [
            document_dir, indices_dir, tmp_audio_dir, chat_history_dir, stock_data_dir, generated_images_dir
        ]:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                log_debug(f"Directory '{directory_path}' created successfully.")

        # Initialize common variables, models
        self.document_dir = document_dir
        self.indices_dir = indices_dir
        self.tmp_audio_dir = tmp_audio_dir
        self.chat_history_dir = chat_history_dir
        self.stock_data_dir = stock_data_dir
        self.generated_images_dir = generated_images_dir
        self.nlp = spacy.load("en_core_web_sm")
        self.ner_labels = self.nlp.get_pipe("ner").labels

    def render_media_processing_page(self):
        """
        Media input and processing page.
        """
        render_media_processing_page(
            document_dir=self.document_dir, tmp_audio_dir=self.tmp_audio_dir, audio_model=self.openai_wisper_util,
            reddit_util=self.reddit_util
        )

    def render_manage_index_page(self):
        """
        Create/Manage/Delete document index page.
        """
        render_manage_index_page(document_dir=self.document_dir, indices_dir=self.indices_dir)

    def render_document_explore_page(self):
        """
        Document explore page.
        """
        render_document_explore_page(
            document_dir=self.document_dir, indices_dir=self.indices_dir, nlp=self.nlp, ner_labels=self.ner_labels
        )

    def render_qa_page(self, temperature=None, max_tokens=None, model_name=None, embedding_model_name=None, enable_tts=False, tts_voice=None):
        """
        Question answer page.
        """
        render_qa_page(
            temperature=temperature, max_tokens=max_tokens, model_name=model_name,
            embedding_model_name=embedding_model_name, tx2sp_util=self.openai_t2s_util,
            indices_dir=self.indices_dir,  chat_history_dir=self.chat_history_dir, enable_tts=enable_tts,
            tts_voice=tts_voice
        )

    def render_stocks_comparison_page(self):
        """
        Stocks comparison page.
        """
        render_stocks_comparison_page(stock_data_dir=self.stock_data_dir)

    def render_image_generation_page(self):
        """
        Image generation page.
        """
        render_image_generation_page(
            generated_images_dir=self.generated_images_dir, image_generation_util=self.openai_dalle_util
        )


def main():
    APP_NAME = "VerbalVista"
    APP_VERSION = "1.3"
    APP_PAGES = [
        "Media Processing", "Explore Document", "Manage Index", "Q & A", "Stocks Comparison",
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
    stock_data_dir = 'data/stock_data_dir/'
    generated_images_dir = 'data/generated_images/'

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
        stock_data_dir=stock_data_dir, generated_images_dir=generated_images_dir
    )
    if selected_page == "Media Processing":
        vv.render_media_processing_page()
    elif selected_page == "Manage Index":
        vv.render_manage_index_page()
    elif selected_page == "Q & A":
        with st.sidebar:
            temperature = st.number_input("Temperature", value=0.5, min_value=0.0, max_value=1.0)
            max_tokens = st.number_input("Max Tokens", value=512, min_value=0, max_value=4000)
            model_name = st.selectbox("Model Name", ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k", "gpt-4-1106-preview"], index=4)
            embedding_model_name = st.selectbox("Embedding Model Name", ["text-embedding-ada-002"], index=0)
            enable_tts = st.checkbox("Enable text-to-speech", value=False)
            tts_voice = "echo"
            if enable_tts:
                tts_voice = st.selectbox("Select Voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], index=1)
        vv.render_qa_page(
            temperature=temperature, max_tokens=max_tokens, model_name=model_name,
            embedding_model_name=embedding_model_name, enable_tts=enable_tts, tts_voice=tts_voice
        )
    elif selected_page == "Explore Document":
        vv.render_document_explore_page()
    elif selected_page == 'Stocks Comparison':
        vv.render_stocks_comparison_page()
    elif selected_page == 'Image Generation':
        vv.render_image_generation_page()


if __name__ == '__main__':
    main()
