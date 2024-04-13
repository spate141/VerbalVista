import os
import hmac
import streamlit as st
from app_pages import *
from dotenv import load_dotenv; load_dotenv()

from utils import log_info, log_debug, log_error
from utils.rag_utils import LLM_MAX_CONTEXT_LENGTHS, EMBEDDING_DIMENSIONS
from utils.data_parsing_utils.reddit_comment_parser import RedditSubmissionCommentsFetcher
from utils.openai_utils import OpenAIDalleUtil, OpenAIWisperUtil, OpenAIText2SpeechUtil, OpenAIGPT4ImageAnalysisUtil


def check_password():
    """
    Returns `True` if the user had the correct password.
    """
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.markdown("<h2>🔐 Enter your password:</h2>", unsafe_allow_html=True)
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password", label_visibility='collapsed'
    )
    if "password_correct" in st.session_state:
        st.error("❗️Password Incorrect ❗️")
    return False


class VerbalVista:

    def __init__(
        self, document_dir: str = None, tmp_dir: str = None, indices_dir: str = None,
        chat_history_dir: str = None, stock_data_dir: str = None, generated_images_dir: str = None
    ):
        """
        Initializes the VerbalVista application with directories for storing documents, temporary files,
        indices, chat history, stock data, and generated images. It also initializes utilities for
        Whisper (speech-to-text), Text-to-Speech, DALL-E (image generation), and fetching comments from Reddit posts.

        :param document_dir: Directory for storing documents.
        :param tmp_dir: Directory for storing temporary files.
        :param indices_dir: Directory for storing indices for document search.
        :param chat_history_dir: Directory for storing chat histories.
        :param stock_data_dir: Directory for storing stock data.
        :param generated_images_dir: Directory for storing images generated by DALL-E.
        """
        # Initialize all necessary classes
        self.openai_wisper_util = OpenAIWisperUtil(api_key=os.getenv("OPENAI_API_KEY"))
        self.openai_t2s_util = OpenAIText2SpeechUtil(api_key=os.getenv("OPENAI_API_KEY"))
        self.openai_dalle_util = OpenAIDalleUtil(api_key=os.getenv("OPENAI_API_KEY"))
        self.openai_img_understanding_util = OpenAIGPT4ImageAnalysisUtil(api_key=os.getenv("OPENAI_API_KEY"))
        self.reddit_util = RedditSubmissionCommentsFetcher(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )

        # Create relevant directories
        for directory_path in [
            document_dir, indices_dir, tmp_dir, chat_history_dir, stock_data_dir, generated_images_dir
        ]:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                log_debug(f"Directory '{directory_path}' created successfully.")

        # Initialize common variables, models
        self.document_dir = document_dir
        self.indices_dir = indices_dir
        self.tmp_dir = tmp_dir
        self.chat_history_dir = chat_history_dir
        self.stock_data_dir = stock_data_dir
        self.generated_images_dir = generated_images_dir

    def render_media_processing_page(self):
        """
        Media input and processing page.
        """
        render_media_processing_page(
            document_dir=self.document_dir, tmp_dir=self.tmp_dir,
            openai_wisper_util=self.openai_wisper_util, reddit_util=self.reddit_util
        )

    def render_manage_index_page(self):
        """
        Create/Manage/Delete document index page.
        """
        render_manage_index_page(
            document_dir=self.document_dir, indices_dir=self.indices_dir,
            embedding_models=list(EMBEDDING_DIMENSIONS.keys())
        )

    def render_document_explore_page(self):
        """
        Document explore page.
        """
        render_document_explore_page(
            document_dir=self.document_dir, indices_dir=self.indices_dir
        )

    def render_qa_page(
        self, temperature=None, max_tokens=None, model_name=None, enable_tts=False, tts_voice=None,
        max_semantic_retrieval_chunks=None, max_lexical_retrieval_chunks=None
    ):
        """
        Renders the question and answer page, allowing users to ask questions and receive answers based on the data.

        :param temperature: Sampling temperature for language model.
        :param max_tokens: Maximum number of tokens for the language model response.
        :param model_name: Name of the language model to use.
        :param enable_tts: Flag to enable text-to-speech functionality.
        :param tts_voice: Voice option for text-to-speech.
        :param max_semantic_retrieval_chunks: Maximum number of chunks for semantic retrieval.
        :param max_lexical_retrieval_chunks: Maximum number of chunks for lexical retrieval.
        """
        render_qa_page(
            temperature=temperature, max_tokens=max_tokens, model_name=model_name,
            max_semantic_retrieval_chunks=max_semantic_retrieval_chunks,
            max_lexical_retrieval_chunks=max_lexical_retrieval_chunks,
            tx2sp_util=self.openai_t2s_util, indices_dir=self.indices_dir,
            chat_history_dir=self.chat_history_dir, enable_tts=enable_tts, tts_voice=tts_voice
        )

    def render_stocks_comparison_page(self):
        """
        Stocks comparison page.
        """
        render_stocks_comparison_page(stock_data_dir=self.stock_data_dir)

    def render_stocks_portfolio_page(self):
        """
        Stocks portfolio page.
        """
        render_stocks_portfolio_page()

    def render_image_understanding_page(self):
        """
        Image understanding page.
        """
        render_image_understanding_page(
            generated_images_dir=self.generated_images_dir,
            image_generation_util=self.openai_dalle_util,
            image_understanding_util=self.openai_img_understanding_util
        )


def main():
    APP_NAME = "VerbalVista"
    APP_VERSION = "3.3"
    APP_PAGES = [
        "Media Processing", "Explore Document", "Manage Index", "Q & A", "Imagination!",
        "Stocks Comparison", "Stocks Portfolio",
    ]

    # Check password
    if not check_password():
        st.stop()

    # Render sidebar
    selected_page = render_sidebar(
        app_name=APP_NAME, app_version=APP_VERSION, app_pages=APP_PAGES
    )

    # Project local cache directories
    document_dir = 'data/documents/'
    tmp_dir = 'data/tmp_dir/'
    indices_dir = 'data/indices/'
    chat_history_dir = 'data/chat_history/'
    stock_data_dir = 'data/stock_data_dir/'
    generated_images_dir = 'data/generated_images/'

    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("ANTHROPIC_API_KEY"):
        # if both env variable and explicit key is not set
        st.error("OpenAI and Anthropic API keys not found!")
        log_error("No OpenAI/Anthropic key found!")

    else:
        vv = VerbalVista(
            document_dir=document_dir, indices_dir=indices_dir,
            tmp_dir=tmp_dir, chat_history_dir=chat_history_dir,
            stock_data_dir=stock_data_dir, generated_images_dir=generated_images_dir
        )
        if selected_page == "Media Processing":
            vv.render_media_processing_page()
        elif selected_page == "Manage Index":
            with st.sidebar:
                with st.expander("Check Model Usage", expanded=True):
                    st.markdown(
                        '- [Anthropic](https://console.anthropic.com/settings/usage)\n'
                        '- [OpenAI](https://platform.openai.com/usage)'
                    )
            vv.render_manage_index_page()
        elif selected_page == "Q & A":
            with st.sidebar:
                with st.expander("Modify LLM Setting"):
                    temperature = st.number_input("Temperature", value=0.5, min_value=0.0, max_value=1.0)
                    max_tokens = st.number_input("Max Tokens", value=512, min_value=0, max_value=4000)
                    max_semantic_retrieval_chunks = st.number_input("Max Semantic Chunks", value=5, min_value=1, max_value=9999999)
                    max_lexical_retrieval_chunks = st.number_input("Max Lexical Chunks", value=1, min_value=1, max_value=9999999)
                    model_name = st.selectbox("Model Name", list(LLM_MAX_CONTEXT_LENGTHS.keys()), index=6)
                    enable_tts = st.checkbox("Enable text-to-speech", value=False)
                    tts_voice = "echo"
                    if enable_tts:
                        tts_voice = st.selectbox("Select Voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], index=1)
                with st.expander("Check Model Usage", expanded=True):
                    st.markdown(
                        '- [Anthropic](https://console.anthropic.com/settings/usage)\n'
                        '- [OpenAI](https://platform.openai.com/usage)'
                    )
            vv.render_qa_page(
                temperature=temperature, max_tokens=max_tokens, model_name=model_name,
                enable_tts=enable_tts, tts_voice=tts_voice,
                max_semantic_retrieval_chunks=max_semantic_retrieval_chunks,
                max_lexical_retrieval_chunks=max_lexical_retrieval_chunks
            )
        elif selected_page == "Explore Document":
            vv.render_document_explore_page()
        elif selected_page == 'Stocks Comparison':
            vv.render_stocks_comparison_page()
        elif selected_page == 'Stocks Portfolio':
            vv.render_stocks_portfolio_page()
        elif selected_page == 'Imagination!':
            with st.sidebar:
                with st.expander("Check Model Usage", expanded=True):
                    st.markdown(
                        '- [Anthropic](https://console.anthropic.com/settings/usage)\n'
                        '- [OpenAI](https://platform.openai.com/usage)'
                    )
            vv.render_image_understanding_page()


if __name__ == '__main__':
    main()
