import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from typing import Optional, Dict, List, Union
from utils.rag_utils.embedding_util import get_embedding_client


def get_query_embedding(query: str, embedding_model_name: Optional[str] = None) -> np.ndarray:
    """
    Retrieves the embedding vector for a given query string using a specified embedding model.
    The resulting query embedding is normalized to unit length.

    :param query: The query string to be embedded.
    :param embedding_model_name: The name of the embedding model to use. If None, a default model is selected.
    :return: A normalized numpy array representing the query's embedding vector.
    """
    embedding_client = get_embedding_client(embedding_model_name)
    response = embedding_client.embeddings.create(input=query, model=embedding_model_name)
    query_emb = np.array(response.data[0].embedding)
    norm = np.linalg.norm(query_emb)
    return query_emb / norm if norm > 0 else query_emb


def do_semantic_search(
        query_embedding: np.ndarray, faiss_index: faiss.IndexFlatIP, metadata_dict: Dict[int, Dict[str, str]],
        k: int = 5
) -> List[Dict[str, Union[int, float, str]]]:
    """
    Performs a semantic search using a FAISS index to find the top-k most semantically similar embeddings to a given query embedding.
    Results are enriched with metadata from the metadata dictionary.

    :param query_embedding: The embedding vector of the query.
    :param faiss_index: The FAISS index that contains embeddings to search against.
    :param metadata_dict: A dictionary mapping embedding indices to their metadata.
    :param k: The number of top results to return.
    :return: A list of dictionaries, each representing a search result with id, distance, text, and source.
    """
    # Searching the FAISS index
    D, I = faiss_index.search(np.array([query_embedding]), k)

    # Retrieve results and metadata
    semantic_context = []
    for idx, distance in zip(I[0], D[0]):
        if idx < len(metadata_dict):  # Check if the index is within bounds
            try:
                data = metadata_dict[idx]
                semantic_context.append({
                    "id": idx, "distance": distance, "text": data['text'], "source": data['source']
                })
            except KeyError:
                pass
    return semantic_context


def do_lexical_search(
        lexical_index: BM25Okapi, query: str, metadata_dict: Dict[int, Dict[str, str]], k: int
) -> List[Dict[str, Union[int, str, float]]]:
    """
    Conducts a lexical search for a given query against indexed documents, returning the top-k most relevant results.
    Results are based on BM25 scoring.

    :param lexical_index: The BM25Okapi index containing document tokens for lexical search.
    :param query: The query string to search for.
    :param metadata_dict: A dictionary mapping document indices to their metadata.
    :param k: The number of top results to return.
    :return: A list of dictionaries, each representing a search result with id, text, source, and BM25 score.
    """
    # preprocess query
    query_tokens = query.lower().split()

    # get best matching (BM) scores
    scores = lexical_index.get_scores(query_tokens)

    # sort and get top k
    indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    lexical_context = [{
        "id": i,
        "text": metadata_dict[i]['text'],
        "source": metadata_dict[i]['source'],
        "score": scores[i]
    } for i in indices]
    return lexical_context


def do_no_search(
        metadata_dict: Dict[int, Dict[str, str]]
) -> List[Dict[str, Union[int, str, float]]]:
    """
    Do no search and just return all the content.
    """
    all_context = []
    for index, data_object in metadata_dict.items():
        all_context.append({
            "id": index,
            "text": data_object['text'],
            "source": data_object['source']
        })
    return all_context

