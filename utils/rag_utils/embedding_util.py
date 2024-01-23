import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from utils import log_info, log_debug, log_error


def get_embedding_model(embedding_model_name, model_kwargs=None, encode_kwargs=None):
    """
    Given the embedding_model_name; this will either use the OpenAI API or
    download the model with HuggingFaceEmbeddings.
    :param embedding_model_name: Model name, could also be model_path
    :param model_kwargs: Model kwargs (i.e. {"device": "cuda"})
    :param encode_kwargs: Encoding kwargs (i.e. {"device": "cuda", "batch_size": 100})
    :return embedding model class instance
    """
    if embedding_model_name == "text-embedding-ada-002":
        embedding_model = OpenAIEmbeddings(
            model=embedding_model_name
        )
    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,  # also works with model_path
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
    return embedding_model


class EmbedChunks:

    def __init__(self, model_name):
        self.embedding_model = get_embedding_model(
            embedding_model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 100}
        )

    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        log_debug(f"Generated embeddings shape: {np.array(embeddings).shape}")
        return {"text": batch["text"], "source": batch["source"], "embeddings": embeddings}

