import os
import time
import ray
import glob
import faiss
import pickle
import pandas as pd
from pathlib import Path
from functools import partial
from ray.data import ActorPoolStrategy
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import log_info, log_error, log_debug
from utils.rag_utils.agent_util import QueryAgent
from utils.rag_utils.indexing_util import StoreResults, FaissIndexActor
from utils.rag_utils.embedding_util import EmbedChunks, EMBEDDING_DIMENSIONS


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


def do_some_data_extraction(record):
    """
    Extract data from the record['path'] txt file and return [{source: filename, text: text}, ...] object
    """
    filename = record["path"].name
    with open(record["path"], "r", encoding="utf-8") as f:
        text = f.read()
    data = [{"source": filename, "text": text}]
    return data


def do_some_data_chunking(data, chunk_size=600, chunk_overlap=30):
    """
    Split the input data into proper chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.create_documents(
        texts=[data["text"]],
        metadatas=[{"source": data["source"]}]
    )
    return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]


def index_data(
        document_directory: str = None, index_directory: str = None, chunk_size: int = 600,
        chunk_overlap: int = 30, embedding_model: str = "text-embedding-ada-002"
):
    """
    This function accepts a document_directory which contain files in plain text format and
    create an index and save that index into index_directory.
    :param document_directory:
    :param index_directory:
    :param chunk_size:
    :param chunk_overlap:
    :param embedding_model:
    """
    # Load meta
    meta_loader = DirectoryLoader(document_directory, glob="**/*.meta.txt")
    try:
        raw_meta = meta_loader.load()[0].page_content
    except IndexError:
        log_error(f"Meta file not found for: {document_directory}")
        raw_meta = ""

    # Load Data
    document_directory = Path(document_directory)
    ds = ray.data.from_items(
        [{"path": path} for path in document_directory.rglob("*.data.txt") if not path.is_dir()]
    )
    text_data_ds = ds.flat_map(do_some_data_extraction)

    # Chunk data
    chunks_ds = text_data_ds.flat_map(
        partial(
            do_some_data_chunking, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    )

    # Embed chunks
    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={"model_name": embedding_model},
        batch_size=100,
        num_gpus=0,
        compute=ActorPoolStrategy(size=1),
    )

    # Initialize the actor
    faiss_actor = FaissIndexActor.remote(EMBEDDING_DIMENSIONS[embedding_model])

    # Process batches
    embedded_chunks.map_batches(
        StoreResults,
        fn_constructor_kwargs={"faiss_actor": faiss_actor},
        batch_size=100,
        num_cpus=1,
        compute=ActorPoolStrategy(size=1),
    ).count()

    # Save the final index and metadata
    if not os.path.exists(index_directory):
        os.makedirs(index_directory)
    ray.get(faiss_actor.save_index.remote(index_directory))

    # Save document meta
    doc_meta_path = os.path.join(index_directory, "doc.meta.txt")
    with open(doc_meta_path, 'w') as f:
        f.write(raw_meta)


def load_index_and_metadata(index_directory: str = None):
    """
    Load FAISS index and metadata dict from local disk.
    :param index_directory: Directory containing FAISS index and metadata dict.
    :return: {"faiss_index": faiss_index, "lexical_index": lexical_index, "metadata_dict": metadata_dict}
    """
    index_path = os.path.join(index_directory, 'faiss.index')
    lexical_index_path = os.path.join(index_directory, 'lexical.index')
    metadata_path = os.path.join(index_directory, 'index.metadata')
    if os.path.exists(index_path) and os.path.exists(metadata_path) and os.path.exists(lexical_index_path):
        log_debug(f'Loading FAISS index, Lexical index and metadata dict from: {index_directory}')
        faiss_index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata_dict = pickle.load(f)
        with open(lexical_index_path, "rb") as f:
            lexical_index = pickle.load(f)
        return {"faiss_index": faiss_index, "lexical_index": lexical_index, "metadata_dict": metadata_dict}
    else:
        log_error(f'No index or metadata found: {index_directory}')
        return {"faiss_index": None, "lexical_index": None, "metadata_dict": None}


def do_some_chat_completion(
        query: str = None, embedding_model: str = "text-embedding-ada-002", llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.5, faiss_index=None, lexical_index=None, metadata_dict=None, reranker=None,
        max_semantic_retrieval_chunks: int = 5, max_lexical_retrieval_chunks: int = 1
):
    query_agent = QueryAgent(
        faiss_index=faiss_index, metadata_dict=metadata_dict, lexical_index=lexical_index, reranker=reranker
    )
    result = query_agent(
        query=query, stream=False, num_chunks=max_semantic_retrieval_chunks,
        lexical_search_k=max_lexical_retrieval_chunks, temperature=temperature,
        embedding_model_name=embedding_model, llm_model=llm_model
    )
    return result
