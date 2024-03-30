import numpy as np
from utils.rag_utils.embedding_util import get_embedding_client


def get_query_embedding(query, embedding_model_name=None):
    """
    Get query vector and return normalized query vector.
    """
    embedding_client = get_embedding_client(embedding_model_name)
    response = embedding_client.embeddings.create(input=query, model=embedding_model_name)
    query_emb = np.array(response.data[0].embedding)
    norm = np.linalg.norm(query_emb)
    return query_emb / norm if norm > 0 else query_emb


def do_semantic_search(query_embedding, faiss_index, metadata_dict, k=5):
    """
    Use FAISS index and search for top-k most similar chunks with query embedding.
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


def do_lexical_search(lexical_index, query, metadata_dict, k):
    """
    Do the lexical search for query and chunks and return top-k most matched chunks
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

