import os
import time
import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from llama_index import StorageContext, load_index_from_storage
from llama_index import SimpleDirectoryReader, VectorStoreIndex, LLMPredictor, PromptHelper
from .logging_module import log_info


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


class VectorIndex:

    def __init__(self):
        pass

    @staticmethod
    def get_available_indices(tmp_indices_dir: str = 'indices/'):
        """

        :param tmp_indices_dir:
        :return:
        """
        indices_data = []
        all_indices_dirs = os.listdir(tmp_indices_dir)
        all_indices_dirs = [i for i in all_indices_dirs if not i.startswith('.')]
        for index_dir in all_indices_dirs:
            index_dir_path = os.path.join(tmp_indices_dir, index_dir)
            creation_date = time.ctime(os.stat(index_dir_path).st_ctime)
            indices_data.append((index_dir_path, creation_date))
        df = pd.DataFrame(indices_data, columns=['Index Name', 'Creation Date'])
        return df

    @staticmethod
    def get_available_documents(tmp_document_dir: str = 'documents/', tmp_indices_dir: str = 'indices/'):
        """

        :param tmp_document_dir:
        :param tmp_indices_dir:
        :return:
        """
        transcripts_data = []
        transcripts_subdirs = next(os.walk(tmp_document_dir))[1]
        indices_subdirs = next(os.walk(tmp_indices_dir))[1]
        for transcript in transcripts_subdirs:
            transcript_path = os.path.join(tmp_document_dir, transcript)
            creation_date = time.ctime(os.stat(transcript_path).st_ctime)
            if transcript in indices_subdirs:
                transcripts_data.append((False, '✅', transcript_path, creation_date))
            else:
                transcripts_data.append((False, '❓', transcript_path, creation_date))
        df = pd.DataFrame(transcripts_data, columns=['Create Index', 'Index Status', 'Document Name', 'Creation Date'])
        return df

    @staticmethod
    def index_document(
            document_directory: str = None, index_directory: str = None, context_window: int = 3900,
            num_outputs: int = 512, chunk_overlap_ratio: float = 0.1, chunk_size_limit: int = 600
    ):
        """
        Indexes the documents in the specified folder using the GPTVectorStoreIndex.

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
        """
        prompt_helper = PromptHelper(
            context_window, num_outputs, chunk_overlap_ratio, chunk_size_limit=chunk_size_limit
        )
        llm_predictor = LLMPredictor(
            llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs)
        )
        documents = SimpleDirectoryReader(document_directory).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            llm_predictor=llm_predictor,
            prompt_helper=prompt_helper
        )
        index.storage_context.persist(persist_dir=index_directory)
