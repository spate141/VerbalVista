import os
import ray
import time
import faiss
import pickle
import tiktoken
import structlog
import numpy as np
from ray import serve
from pathlib import Path
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List
from langchain.embeddings import OpenAIEmbeddings
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Inference API for VerbalVista",
    description="🅛🅛🅜 + Your Data = 🩶",
    version="1.1",
)

MAX_CONTEXT_LENGTHS = {
    'gpt-4': 8192,
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-16k': 16384,
    'meta-llama/Llama-2-7b-chat-hf': 4096,
    'meta-llama/Llama-2-13b-chat-hf': 4096,
    'meta-llama/Llama-2-70b-chat-hf': 4096,
    'codellama/CodeLlama-34b-Instruct-hf': 16384,
    'mistralai/Mistral-7B-Instruct-v0.1': 65536
}

SYS_PROMPT = "Answer the query using the context provided. Be succinct. " \
"Contexts are organized in a list of dictionaries [{'text': <context>}, {'text': <context>}, ...]. " \
"Feel free to ignore any contexts in the list that don't seem relevant to the query. " \
"If the question cannot be answered using the information provided answer with 'I don't know'."

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    query: str


class Answer(BaseModel):
    question: str
    answer: str
    sources: List[str]
    llm: str


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

        faiss_index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata_dict = pickle.load(f)
        with open(lexical_index_path, "rb") as f:
            lexical_index = pickle.load(f)
        return {"faiss_index": faiss_index, "lexical_index": lexical_index, "metadata_dict": metadata_dict}
    else:

        return {"faiss_index": None, "lexical_index": None, "metadata_dict": None}


def get_embedding_model(embedding_model_name, model_kwargs=None, encode_kwargs=None):
    """
    Given the embedding_model_name; this will either use the OpenAI API or
    download the model with HuggingFaceEmbeddings.
    :param embedding_model_name: Model name, could also be model_path
    :param model_kwargs: Model kwargs (i.e. {"device": "cuda"})
    :param encode_kwargs: Encoding kwargs (i.e. {"device": "cuda", "batch_size": 100})
    :return embedding model class instance
    """
    embedding_model = OpenAIEmbeddings(model=embedding_model_name)
    return embedding_model


def get_query_embedding(query, embedding_model_name=None):
    """
    Get query vector and return normalized query vector.
    """
    embedding_model = get_embedding_model(embedding_model_name)
    query_emb = np.array(embedding_model.embed_query(query))
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
            data = metadata_dict[idx]
            semantic_context.append({"id": idx, "distance": distance, "text": data['text'], "source": data['source']})
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


def get_num_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


class QueryAgent:

    def __init__(
            self, embedding_model_name="text-embedding-ada-002",  llm_model="gpt-3.5-turbo",
            temperature=0.5, max_context_length=4096, system_content=SYS_PROMPT,
            faiss_index=None, metadata_dict=None, lexical_index=None, reranker=None
    ):

        # Embedding model for query encoding
        self.embedding_model_name = embedding_model_name

        # Context length (restrict input length to 50% of total context length)
        try:
            max_context_length = int(0.5 * MAX_CONTEXT_LENGTHS[llm_model])
        except KeyError:
            max_context_length = int(0.5 * max_context_length)

        # Lexical search
        self.lexical_index = lexical_index

        # Re-ranker
        self.reranker = reranker

        # LLM
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_model = llm_model
        self.temperature = temperature
        self.context_length = max_context_length - get_num_tokens(system_content)
        self.system_content = system_content

        # Vectorstore
        self.faiss_index = faiss_index
        self.metadata_dict = metadata_dict

    @staticmethod
    def trim(text, max_context_length):
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.decode(enc.encode(text)[:max_context_length])

    @staticmethod
    def response_stream(response):
        for chunk in response:
            yield chunk.choices[0].delta.content

    def prepare_response(self, response, stream):
        if stream:
            return self.response_stream(response)
        else:
            return response.choices[-1].message.content

    def generate_response(
            self, llm_model, temperature=0.5, stream=True, system_content="", user_content="",
            max_retries=1, retry_interval=60
    ):
        """Generate response from an LLM."""
        retry_count = 0
        while retry_count <= max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=llm_model,
                    temperature=temperature,
                    stream=stream,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                )
                return self.prepare_response(response=completion, stream=stream)

            except Exception as e:
                print(f"Exception: {e}")
                time.sleep(retry_interval)  # default is per-minute rate limits
                retry_count += 1
        return ""

    def __call__(self, query, num_chunks=5, stream=True, lexical_search_k=1, rerank_threshold=0.2, rerank_k=7):

        # Get sources and context
        query_embedding = get_query_embedding(query, embedding_model_name=self.embedding_model_name)

        # {id, distance, text, source}
        context_results = do_semantic_search(
            query_embedding, self.faiss_index, self.metadata_dict, k=num_chunks
        )

        # Add lexical search results
        if self.lexical_index:
            lexical_context = do_lexical_search(
                self.lexical_index, query, self.metadata_dict, lexical_search_k
            )
            # Insert after <lexical_search_k> worth of semantic results
            context_results[lexical_search_k:lexical_search_k] = lexical_context

        # Rerank
        if self.reranker:
            pass

        # Generate response
        context = [{"text": item["text"]} for item in context_results]
        sources = [item["source"] for item in context_results]
        user_content = f"query: {query}, context: {context}"
        answer = self.generate_response(
            llm_model=self.llm_model,
            temperature=self.temperature,
            stream=stream,
            system_content=self.system_content,
            user_content=self.trim(user_content, self.context_length)
        )

        # Result
        result = {"question": query, "sources": sources, "answer": answer, "llm": self.llm_model}
        return result


@serve.deployment(
    num_replicas=1, ray_actor_options={"num_cpus": 4, "num_gpus": 0}
)
@serve.ingress(app)
class RayAssistantDeployment:

    def __init__(
            self, index_directory="/Users/snehal/PycharmProjects/VerbalVista/data/indices/www-youtube-com-watch?v=gL6Vzt8FYS0",
            embedding_model_name="text-embedding-ada-002", llm_model_name="gpt-4", temperature=0.5
    ):
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
        self.logger = structlog.get_logger()

        # get faiss, lexical and metadata
        index_meta = load_index_and_metadata(index_directory=index_directory)
        # {"faiss_index": faiss_index, "lexical_index": lexical_index, "metadata_dict": metadata_dict}
        faiss_index = index_meta["faiss_index"]
        lexical_index = index_meta["lexical_index"]
        metadata_dict = index_meta["metadata_dict"]

        # get query agent
        self.query_agent = QueryAgent(
            embedding_model_name=embedding_model_name, llm_model=llm_model_name, temperature=temperature,
            faiss_index=faiss_index, metadata_dict=metadata_dict, lexical_index=lexical_index, reranker=None
        )

    def predict(self, query: Query, stream: bool) -> Dict[str, Any]:
        result = self.query_agent(query=query.query, stream=stream)
        return result

    @app.post("/query")
    def query(self, query: Query) -> Answer:
        result = self.predict(query, stream=False)
        return Answer.parse_obj(result)

    def produce_streaming_answer(self, query, result):
        answer = []
        for answer_piece in result["answer"]:
            answer.append(answer_piece)
            yield answer_piece

        if result["sources"]:
            yield "\n\n**Sources:**\n"
            for source in result["sources"]:
                yield "* " + source + "\n"

        self.logger.info(
            "finished streaming query",
            query=query,
            document_ids=result["document_ids"],
            llm=result["llm"],
            answer="".join(answer),
        )

    @app.post("/stream")
    def stream(self, query: Query) -> StreamingResponse:
        result = self.predict(query, stream=True)
        return StreamingResponse(
            self.produce_streaming_answer(query.query, result), media_type="text/plain"
        )


deployment = RayAssistantDeployment.bind(
    index_directory="/Users/snehal/PycharmProjects/VerbalVista/data/indices/www-youtube-com-watch?v=gL6Vzt8FYS0",
    embedding_model_name="text-embedding-ada-002", llm_model_name="gpt-4", temperature=0.5
)
serve.run(deployment, route_prefix="/")

# ray start --head
# python serve.py
# ray stop