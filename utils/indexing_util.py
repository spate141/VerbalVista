import os
import time
import glob
import pandas as pd
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

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
        df = pd.DataFrame(indices_data, columns=['Index Path', 'Creation Date'])
        df['Creation Date'] = pd.to_datetime(df['Creation Date'])
        df = df.sort_values(by='Creation Date', ascending=False)
        return df

    @staticmethod
    def get_available_documents(document_dir: str = 'documents/', indices_dir: str = 'indices/'):
        """

        :param document_dir:
        :param indices_dir:
        :return:
        """
        documents_data = []
        documents_subdirs = next(os.walk(document_dir))[1]
        indices_subdirs = next(os.walk(indices_dir))[1]
        for doc_sub_dir in documents_subdirs:
            document_path = os.path.join(document_dir, doc_sub_dir)
            creation_date = time.ctime(os.stat(document_path).st_ctime)
            try:
                doc_meta_data_path = glob.glob(f"{document_path}/*.meta.txt")[0]
                doc_meta_data = open(doc_meta_data_path, 'r').read()
                doc_meta_data = ' '.join(doc_meta_data.split())
            except IndexError:
                doc_meta_data = None
            if doc_sub_dir in indices_subdirs:
                documents_data.append((False, '✅', doc_meta_data, document_path, creation_date))
            else:
                documents_data.append((False, '❓', doc_meta_data, document_path, creation_date))
        df = pd.DataFrame(
            documents_data,
            columns=['Select Index', 'Index Status', 'Document Meta', 'Document Name', 'Creation Date']
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
        data_loader = DirectoryLoader(document_directory, glob="**/*.data.txt")
        meta_loader = DirectoryLoader(document_directory, glob="**/*.meta.txt")

        raw_documents = data_loader.load()

        try:
            raw_meta = meta_loader.load()[0].page_content
        except IndexError:
            log_error(f"Meta file not found for: {document_directory}")
            raw_meta = None

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=20, length_function=len,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        documents = text_splitter.split_documents(raw_documents)

        # Load Data to vectorstore
        with get_openai_callback() as cb:
            embeddings = OpenAIEmbeddings(model=embedding_model, chunk_size=chunk_size)
            vectorstore = FAISS.from_documents(documents, embeddings)

        if not os.path.exists(index_directory):
            os.makedirs(index_directory)

        # Save vectorstore
        faiss_index_path = os.path.join(index_directory, 'faiss')
        vectorstore.save_local(faiss_index_path)

        # Save document meta
        doc_meta_path = os.path.join(index_directory, "doc.meta.txt")
        with open(doc_meta_path, 'w') as f:
            f.write(raw_meta)

        return cb
