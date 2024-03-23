import os
import time
import ray
import glob
import faiss
import pickle
import pandas as pd
from typing import Dict
from pathlib import Path
from functools import partial
from ray.data import ActorPoolStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import log_info, log_error, log_debug
from utils.rag_utils import EMBEDDING_DIMENSIONS, MODEL_COST_PER_1K_TOKENS
from utils.rag_utils.agent_util import GPTAgent, ClaudeAgent
from utils.rag_utils.indexing_util import StoreResults, FaissIndexActor
from utils.rag_utils.embedding_util import EmbedChunks


def get_available_indices(indices_dir: str = 'indices/'):
    """
    Retrieves a list of available index directories, excluding hidden ones, along with their metadata and creation dates.

    This function scans the specified directory for subdirectories that represent index directories,
    ignoring any that are hidden (start with a dot). It attempts to read a metadata file with a '.meta.txt'
    extension within each index directory to gather descriptive information about the index. It then collects
    the path to the index directory, its metadata, and its creation date, returning this information as a
    sorted pandas DataFrame.

    :param indices_dir: The path to the directory containing index subdirectories. Defaults to 'indices/'.
                        The trailing slash indicates it is a directory.
    :return: A pandas DataFrame with columns ['Index Path', 'Index Description', 'Creation Date'],
             sorted by 'Creation Date' in descending order.
    """
    indices_data = []
    all_indices_dirs = os.listdir(indices_dir)
    all_indices_dirs = [i for i in all_indices_dirs if not i.startswith('.')]
    for index_dir in all_indices_dirs:
        index_dir_path = os.path.join(indices_dir, index_dir)
        creation_date = time.ctime(os.stat(index_dir_path).st_ctime)
        try:
            index_meta_data_path = glob.glob(f"{index_dir_path}/*.meta.txt")[0]
            index_meta_data = open(index_meta_data_path, 'r').read()
            index_meta_data = ' '.join(index_meta_data.split())
        except IndexError:
            index_meta_data = None
        indices_data.append((index_dir_path, index_meta_data, creation_date))
    df = pd.DataFrame(indices_data, columns=['Index Path', 'Index Description', 'Creation Date'])
    df['Creation Date'] = pd.to_datetime(df['Creation Date'])
    df = df.sort_values(by='Creation Date', ascending=False)
    return df


def count_files_in_dir(dir_path: str = None):
    """
    Counts the number of '.data.txt' files in a given directory.

    This function checks if the provided directory path exists and is indeed a directory. If it is,
    the function counts the number of files within that directory that have a '.data.txt' extension.

    :param dir_path: The path to the directory where files will be counted.
    :return: An integer representing the count of '.data.txt' files in the specified directory.
             Returns 0 if the directory does not exist or the path does not refer to a directory.
    """
    if not os.path.isdir(dir_path):
        return 0
    return len([
        f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.data.txt')
    ])


def get_available_documents(document_dir: str = 'documents/', indices_dir: str = 'indices/'):
    """
    Gathers information about document directories and their indexing status.

    This function scans the provided document directory for subdirectories, then checks each one for a metadata file
    and counts the number of '.data.txt' files. It also compares these document subdirectories against index
    subdirectories to determine whether each document directory has been indexed. The collected data is returned as a
    pandas DataFrame with relevant information such as indexing status, metadata, directory name, total number of
    files, and creation date.

    :param document_dir: The path to the directory containing document subdirectories. Defaults to 'documents/'.
    :param indices_dir: The path to the directory containing index subdirectories. Defaults to 'indices/'.
    :return: A pandas DataFrame with columns ['Select Index', 'Index Status', 'Document Meta', 'Directory Name',
             'Total Files', 'Creation Date'] representing the gathered information about each document subdirectory.
    """
    documents_data = []
    documents_subdirs = next(os.walk(document_dir))[1]
    indices_subdirs = next(os.walk(indices_dir))[1]
    for doc_sub_dir in documents_subdirs:
        subdir_path = os.path.join(document_dir, doc_sub_dir)
        creation_date = time.ctime(os.stat(subdir_path).st_ctime)
        total_files = count_files_in_dir(subdir_path)
        try:
            doc_meta_data_path = glob.glob(f"{subdir_path}/*.meta.txt")[0]
            doc_meta_data = open(doc_meta_data_path, 'r').read()
            doc_meta_data = ' '.join(doc_meta_data.split())
        except IndexError:
            doc_meta_data = None
        if doc_sub_dir in indices_subdirs:
            documents_data.append((False, '✅', doc_meta_data, subdir_path, total_files, creation_date))
        else:
            documents_data.append((False, '❓', doc_meta_data, subdir_path, total_files, creation_date))
    df = pd.DataFrame(
        documents_data,
        columns=['Select Index', 'Index Status', 'Document Meta', 'Directory Name', 'Total Files', 'Creation Date']
    )
    return df


def delete_directory(selected_directory: str = None):
    """
    Deletes a specified directory along with all its contained files and subdirectories.

    This function recursively traverses the specified directory from the bottom up, deleting all files
    and subdirectories before finally deleting the top-level directory itself. If any operation fails,
    an error is logged with the description of the issue.

    :param selected_directory: The path to the directory that should be deleted. If this parameter
                               is not provided or `None`, the function will not perform any action.
    :return: None
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
        return True
    except OSError as e:
        log_error(f"Error: {e.strerror}")
        return False


def do_some_data_extraction(record: Dict = None):
    """
    Extracts text data from a file specified in the given record.

    This function opens the file located at the path indicated by 'record['path']', reads its content, and returns a
    list of dictionaries. Each dictionary contains two key-value pairs: 'source' with the name of the file, and 'text'
    with the content of the file.

    :param record: A dictionary containing the key 'path' that holds a Path object pointing to the target text file.
    :return: A list containing a single dictionary with 'source' set to the filename and 'text' set to the file content.
    """
    filename = record["path"].name
    with open(record["path"], "r", encoding="utf-8") as f:
        text = f.read()
    data = [{"source": filename, "text": text}]
    return data


def do_some_data_chunking(data, chunk_size=600, chunk_overlap=30):
    """
    Splits the input text data into chunks of a specified size with a defined overlap between consecutive chunks.

    This function utilizes a RecursiveCharacterTextSplitter to divide the provided text data into smaller parts,
    each of a maximum character length defined by 'chunk_size'. The chunks can optionally overlap by a number of
    characters specified by 'chunk_overlap'. The function returns a list of dictionaries, where each dictionary
    represents a text chunk and its source information.

    :param data: A dictionary containing the keys 'text' with the full text to be chunked and 'source' with the
                 source information.
    :param chunk_size: The maximum number of characters for each chunk. Defaults to 600.
    :param chunk_overlap: The number of characters that each chunk will overlap with the next. Defaults to 30.
    :return: A list of dictionaries, each containing a 'text' key with the chunk content and a 'source' key with
             the source information from the input data.
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


def load_first_meta_file(directory):
    """
    Load the content of the first .meta.txt file found in the given directory.

    Args:
    directory (str): Path to the directory to search in.

    Returns:
    str: Content of the first .meta.txt file, or an empty string if none found.
    """
    meta_files = directory.rglob('*.meta.txt')

    for meta_file in meta_files:
        try:
            with open(meta_file, 'r') as file:
                return file.read()
        except Exception as e:
            log_error(f"Error reading file {meta_file}: {e}")

    log_error(f"Meta file not found for: {directory}")
    return ""


def index_data(
    document_directory: str = None, index_directory: str = None, chunk_size: int = 600,
    chunk_overlap: int = 30, embedding_model: str = "text-embedding-3-small"
):
    """
    Indexes text data from a specified document directory and stores the index in an index directory.

    This function processes text files from the document_directory by extracting their content, chunking them into
    smaller parts, embedding these chunks using a specified model, and then storing the results in a FAISS index.
    The index is saved to the index_directory. Metadata is also loaded and saved alongside the index.

    :param document_directory: The path to the directory containing the text files to be indexed.
    :param index_directory: The path where the generated index will be stored.
    :param chunk_size: The number of characters each text chunk should contain. Defaults to 600.
    :param chunk_overlap: The number of characters that should overlap between consecutive chunks. Defaults to 30.
    :param embedding_model: The name of the model used for embedding the text chunks. Defaults to "text-embedding-3-small".
    """
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    document_directory = Path(document_directory)

    # Load meta normally as there will be only one meta file
    raw_meta = load_first_meta_file(document_directory)

    # Load data with ray as there could be multiple data files
    ds = ray.data.from_items(
        [{"path": path} for path in document_directory.rglob("*.data.txt") if not path.is_dir()]
    )
    text_data_ds = ds.flat_map(do_some_data_extraction)

    # Chunk data
    chunks_ds = text_data_ds.flat_map(
        partial(
            do_some_data_chunking, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ),
        num_cpus=1,
        num_gpus=0
    )

    # Embed chunks
    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={"model_name": embedding_model},
        batch_size=100,
        num_cpus=1,
        concurrency=1,
    )

    # Initialize the actor
    faiss_actor = FaissIndexActor.remote(EMBEDDING_DIMENSIONS[embedding_model])

    # Process batches
    embedded_chunks.map_batches(
        StoreResults,
        fn_constructor_kwargs={"faiss_actor": faiss_actor},
        batch_size=100,
        num_cpus=1,
        concurrency=1,
    ).count()

    # Save the final index and metadata
    if not os.path.exists(index_directory):
        os.makedirs(index_directory)
    ray.get(faiss_actor.save_index.remote(index_directory))
    ray.shutdown()

    # Save document meta
    doc_meta_path = os.path.join(index_directory, "doc.meta.txt")
    with open(doc_meta_path, 'w') as f:
        f.write(raw_meta)


def load_index_and_metadata(index_directory: str = None):
    """
    Load the FAISS index, lexical index, and metadata dictionary from the specified directory.

    This function attempts to read the FAISS index, lexical index, and metadata dictionary from
    the given directory. It requires the presence of three specific files within the directory:
    'faiss.index', 'lexical.index', and 'index.metadata'. If any of these files are missing,
    the function logs an error and returns a dictionary with None values.

    Parameters:
    index_directory (str): The path to the directory containing the index and metadata files.
                           If None, defaults to the current working directory.

    Returns:
    dict: A dictionary containing the following key-value pairs:
          - "faiss_index": The loaded FAISS index object, or None if not found.
          - "lexical_index": The loaded lexical index object, or None if not found.
          - "metadata_dict": The loaded metadata dictionary, or None if not found.
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
    query: str = None, embedding_model: str = "text-embedding-3-small", llm_model: str = "gpt-3.5-turbo",
    temperature: float = 0.5, faiss_index=None, lexical_index=None, metadata_dict=None, reranker=None,
    max_semantic_retrieval_chunks: int = 5, max_lexical_retrieval_chunks: int = 1, max_tokens=512
):
    """
    Perform chat completion using a combination of semantic and lexical retrieval methods.

    This function takes a query and uses the provided FAISS index, lexical index, and metadata dictionary
    to retrieve relevant chunks of information. It then utilizes a language model to generate a response
    based on the query and retrieved information. The function supports temperature control for response
    generation variability and allows specification of the number of chunks to retrieve semantically and lexically.

    Parameters:
    query (str): The input query string for which the chat completion is to be performed.
    embedding_model (str): The name of the embedding model to be used for semantic retrieval. Defaults to "text-embedding-3-small".
    llm_model (str): The name of the large language model to be used for response generation. Defaults to "gpt-3.5-turbo".
    temperature (float): Controls randomness in response generation. Lower values make responses more deterministic. Defaults to 0.5.
    faiss_index: The FAISS index object used for semantic retrieval.
    lexical_index: The lexical index object used for lexical retrieval.
    metadata_dict (dict): A dictionary containing metadata associated with the indexed chunks.
    reranker: An optional reranker object to reorder retrieved results based on relevance.
    max_semantic_retrieval_chunks (int): The maximum number of chunks to retrieve semantically. Defaults to 5.
    max_lexical_retrieval_chunks (int): The maximum number of chunks to retrieve lexically. Defaults to 1.
    max_tokens (int): The maximum numbers of tokens a model should generate an output for.
    Returns:
    The result of the chat completion, which includes the generated response and any other relevant information.
    """
    if 'gpt' in llm_model:
        query_agent = GPTAgent(
            faiss_index=faiss_index, metadata_dict=metadata_dict, lexical_index=lexical_index, reranker=reranker
        )
    elif 'claude' in llm_model:
        query_agent = ClaudeAgent(
            faiss_index=faiss_index, metadata_dict=metadata_dict, lexical_index=lexical_index, reranker=reranker
        )
    else:
        raise ValueError(
            f"Unknown model: {llm_model}. Please provide a valid LLM model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_TOKENS.keys())
        )
    result = query_agent(
        query=query, stream=False, num_chunks=max_semantic_retrieval_chunks,
        lexical_search_k=max_lexical_retrieval_chunks, temperature=temperature,
        embedding_model_name=embedding_model, llm_model=llm_model, max_tokens=max_tokens
    )
    return result
