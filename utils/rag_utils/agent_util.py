import os
import time
import tiktoken
from openai import OpenAI
from langchain_community.callbacks.openai_info import get_openai_token_cost_for_model

from utils import log_error
from utils.rag_utils import MAX_CONTEXT_LENGTHS, SYS_PROMPT
from utils.rag_utils.retrieval_util import get_query_embedding, do_lexical_search, do_semantic_search


class QueryAgent:

    def __init__(self, system_content=SYS_PROMPT, faiss_index=None, metadata_dict=None, lexical_index=None, reranker=None):

        # Lexical search
        self.lexical_index = lexical_index

        # Re-ranker
        self.reranker = reranker

        # LLM
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_content = system_content
        self.encoder = tiktoken.get_encoding("cl100k_base")

        # Vectorstore
        self.faiss_index = faiss_index
        self.metadata_dict = metadata_dict

    def trim(self, text, max_context_length):
        return self.encoder.decode(self.encoder.encode(text)[:max_context_length])

    @staticmethod
    def response_stream(response):
        for chunk in response:
            yield chunk.choices[0].delta.content

    def prepare_response(self, response, stream):
        if stream:
            return self.response_stream(response)
        else:
            return response.choices[-1].message.content

    def get_num_tokens(self, text):
        return len(self.encoder.encode(text))

    def generate_response(
            self, llm_model, temperature=0.5, seed=42, stream=False, system_content="", user_content="",
            max_retries=1, retry_interval=60, embedding_model_name="", sources=None
    ):
        """Generate response from an LLM."""
        retry_count = 0
        while retry_count <= max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=llm_model,
                    seed=seed,
                    temperature=temperature,
                    stream=stream,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                )
                _completion_cost = get_openai_token_cost_for_model(
                    llm_model, completion.usage.completion_tokens, is_completion=True
                )
                _prompt_cost = get_openai_token_cost_for_model(
                    llm_model, completion.usage.prompt_tokens, is_completion=False
                )
                completion_meta = {
                    "llm_model": llm_model,
                    "embedding_model": embedding_model_name,
                    "temperature": temperature,
                    "tokens": {
                        "completion": completion.usage.completion_tokens,
                        "prompt": completion.usage.prompt_tokens,
                        "total": completion.usage.total_tokens,
                    },
                    "cost": {
                        "completion": _completion_cost,
                        "prompt": _prompt_cost,
                        "total": _completion_cost + _prompt_cost
                    },
                    "sources": sources
                }
                return self.prepare_response(response=completion, stream=stream), completion_meta

            except Exception as e:
                log_error(f"Exception: {e}")
                time.sleep(retry_interval)  # default is per-minute rate limits
                retry_count += 1
        return None, None

    def __call__(
            self, query, num_chunks=5, stream=False, lexical_search_k=1, temperature=0.5, seed=42,
            embedding_model_name="text-embedding-ada-002", llm_model="gpt-3.5-turbo"
    ):

        # Get sources and context
        query_embedding = get_query_embedding(query, embedding_model_name=embedding_model_name)

        # {id, distance, text, source}
        context_results = do_semantic_search(
            query_embedding, self.faiss_index, self.metadata_dict, k=num_chunks
        )

        # Add lexical search results
        if self.lexical_index:
            lexical_context = do_lexical_search(self.lexical_index, query, self.metadata_dict, lexical_search_k)
            # Insert after <lexical_search_k> worth of semantic results
            context_results[lexical_search_k:lexical_search_k] = lexical_context

        # Rerank
        if self.reranker:
            pass

        # Generate response
        context = [{"text": item["text"]} for item in context_results]
        sources = [item["source"] for item in context_results]
        user_content = f"query: {query}\n\ncontext: {context}\n\n"
        max_context_length = MAX_CONTEXT_LENGTHS.get(llm_model, 4096)
        context_length = max_context_length - self.get_num_tokens(self.system_content)
        answer, completion_meta = self.generate_response(
            llm_model=llm_model,
            temperature=temperature,
            seed=seed,
            stream=stream,
            system_content=self.system_content,
            user_content=self.trim(user_content, context_length),
            embedding_model_name=embedding_model_name,
            sources=sources
        )
        # Result
        result = {"query": query, "answer": answer, "completion_meta": completion_meta}
        return result

