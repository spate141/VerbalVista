import os
import time
import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from llama_index import StorageContext, load_index_from_storage
from llama_index import SimpleDirectoryReader, VectorStoreIndex, LLMPredictor, PromptHelper
from .logging_module import log_info, log_error


@st.cache_resource
def load_index(index_path):
    """
    Load LangChain Index.
    :param index_path:
    :return:
    """
    log_info(f"Loading index: {index_path}")
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    return query_engine


class MyIndex:

    def __init__(self):
        pass

    @staticmethod
    def get_available_indices(indices_dir: str = 'indices/'):
        """

        :param indices_dir:
        :return:
        """
        indices_data = []
        all_indices_dirs = os.listdir(indices_dir)
        all_indices_dirs = [i for i in all_indices_dirs if not i.startswith('.')]
        for index_dir in all_indices_dirs:
            index_dir_path = os.path.join(indices_dir, index_dir)
            creation_date = time.ctime(os.stat(index_dir_path).st_ctime)
            indices_data.append((index_dir_path, creation_date))
        df = pd.DataFrame(indices_data, columns=['Index Name', 'Creation Date'])
        return df

    @staticmethod
    def get_available_documents(document_dir: str = 'documents/', indices_dir: str = 'indices/'):
        """

        :param document_dir:
        :param indices_dir:
        :return:
        """
        transcripts_data = []
        transcripts_subdirs = next(os.walk(document_dir))[1]
        indices_subdirs = next(os.walk(indices_dir))[1]
        for transcript in transcripts_subdirs:
            transcript_path = os.path.join(document_dir, transcript)
            creation_date = time.ctime(os.stat(transcript_path).st_ctime)
            if transcript in indices_subdirs:
                transcripts_data.append((False, '✅', transcript_path, creation_date))
            else:
                transcripts_data.append((False, '❓', transcript_path, creation_date))
        df = pd.DataFrame(
            transcripts_data,
            columns=['Select Index', 'Index Status', 'Document Name', 'Creation Date']
        )
        return df

    @staticmethod
    def delete_document(index_directory: str = None):
        """

        :param index_directory:
        :return:
        """
        try:
            # Remove all files and subdirectories within the directory
            for root, dirs, files in os.walk(index_directory, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    os.remove(file_path)
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    os.rmdir(dir_path)

            # Remove the top-level directory itself
            os.rmdir(index_directory)
            log_info(f"Directory '{index_directory}' deleted successfully.")
        except OSError as e:
            log_error(f"Error: {e.strerror}")

    @staticmethod
    def index_document(
            document_directory: str = None, index_directory: str = None, context_window: int = 3900,
            num_outputs: int = 512, chunk_overlap_ratio: float = 0.1, chunk_size_limit: int = 600,
            temperature: float = 0.7, model_name: str = "gpt-3.5-turbo"
    ):
        """
        Indexes the documents in the specified folder using the VectorStoreIndex.

        Description:
        The function performs the following steps:
        1. Initializes a PromptHelper object with the specified parameters.
        2. Creates an LLMPredictor instance using the ChatOpenAI class, which is a wrapper around an LLMChain from Langchain.
           The LLM is based on OpenAI's "gpt-3.5-turbo" model.
        3. Loads the documents from the specified folder using the SimpleDirectoryReader.
        4. Indexes the documents using the GPTVectorStoreIndex's from_documents() function,
           providing the LLMPredictor and PromptHelper instances.
        5. Persists the index to storage using the storage_context.persist() function.

        The persisted index can be later queried without re-indexing.
        :param document_directory:
        :param index_directory:
        :param context_window:
        :param num_outputs:
        :param chunk_overlap_ratio:
        :param chunk_size_limit:
        :param temperature:
        :param model_name:
        """
        prompt_helper = PromptHelper(
            context_window=context_window, num_output=num_outputs, chunk_overlap_ratio=chunk_overlap_ratio,
            chunk_size_limit=chunk_size_limit
        )
        llm_predictor = LLMPredictor(
            llm=ChatOpenAI(temperature=temperature, model_name=model_name, max_tokens=num_outputs)
        )
        documents = SimpleDirectoryReader(document_directory).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            llm_predictor=llm_predictor,
            prompt_helper=prompt_helper
        )
        index.storage_context.persist(persist_dir=index_directory)

    @staticmethod
    def generate_answer(prompt=None, selected_index_path=None):
        """

        :param prompt:
        :param selected_index_path:
        :return:
        """
        start = time.time()
        query_engine = load_index(selected_index_path)
        with get_openai_callback() as cb:
            response = query_engine.query(prompt)
        total_time = round(time.time() - start, 2)
        response_meta = f"""
            ```markdown
            
            - Total Tokens: {cb.total_tokens}
                - Prompt: {cb.prompt_tokens} + Response: {cb.completion_tokens}
            - Total Cost (USD): ${round(cb.total_cost, 4)}
            - Total Time (Seconds): {total_time}
            
            ```
            """
        log_info({"request_tokens": cb.total_tokens, "request_cost": round(cb.total_cost, 4)})
        return response, response_meta
