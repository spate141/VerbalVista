import os
import time
import tiktoken
from openai import OpenAI

from utils.logging_module import log_info, log_debug, log_error
from utils.retrieval_util import get_query_embedding, do_lexical_search, do_semantic_search


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


def get_num_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


class QueryAgent:

    def __init__(
            self, embedding_model_name="text-embedding-ada-002",  llm_model="gpt-3.5-turbo",
            temperature=0.5, max_context_length=4096, system_content=SYS_PROMPT,
            faiss_index=None, metadata_dict=None, lexical_index=None, chunks=None, reranker=None
    ):

        # Embedding model for query encoding
        self.embedding_model_name = embedding_model_name

        # Context length (restrict input length to 50% of total context length)
        try:
            max_context_length = int(0.5 * MAX_CONTEXT_LENGTHS[llm_model])
        except KeyError:
            log_error(f'{llm_model} NOT FOUND IN `QueryAgent`')
            max_context_length = int(0.5 * max_context_length)

        # Lexical search
        self.chunks = chunks
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
            # predicted_tag = custom_predict(
            #     inputs=[query], classifier=self.reranker, threshold=rerank_threshold
            # )[0]
            # if predicted_tag != "other":
            #     sources = [item["source"] for item in context_results]
            #     reranked_indices = get_reranked_indices(sources, predicted_tag)
            #     context_results = [context_results[i] for i in reranked_indices]
            # context_results = context_results[:rerank_k]
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

