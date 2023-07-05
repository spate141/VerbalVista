import os
import time
import pickle
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from .logging_module import log_info, log_error


class IndexUtil:

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
    def delete_document(selected_directory: str = None):
        """

        :param selected_directory:
        :return:
        """
        try:
            # Remove all files and subdirectories within the directory
            for root, dirs, files in os.walk(selected_directory, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    os.remove(file_path)
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    os.rmdir(dir_path)

            # Remove the top-level directory itself
            os.rmdir(selected_directory)
            log_info(f"Directory '{selected_directory}' deleted successfully.")
        except OSError as e:
            log_error(f"Error: {e.strerror}")

    @staticmethod
    def index_document(
            document_directory: str = None, index_directory: str = None, chunk_size: int = 600,
            embedding_model: str = "text-embedding-ada-002"
    ):
        """
        This function accepts a document_directory which contain files in plain text format and
        create an index and save that index into index_directory.
        :param document_directory:
        :param index_directory:
        :param chunk_size:
        :param embedding_model:
        """
        # Load Data
        loader = DirectoryLoader(document_directory)
        raw_documents = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20, length_function=len)
        documents = text_splitter.split_documents(raw_documents)

        # Load Data to vectorstore
        embeddings = OpenAIEmbeddings(model=embedding_model, chunk_size=chunk_size)
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Save vectorstore
        if not os.path.exists(index_directory):
            os.makedirs(index_directory)

        with open(os.path.join(index_directory, 'vectorstore.pkl'), "wb") as f:
            pickle.dump(vectorstore, f)
