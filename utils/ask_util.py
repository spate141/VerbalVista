import os
import time
import pickle
from typing import Dict, List, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from .logging_module import log_info, log_debug, log_error


class AskUtil:

    def __init__(self):
        pass

    @staticmethod
    def load_vectors(index_directory: str = None, embedding_model: str = "text-embedding-ada-002", chunk_size: int = 600):
        """

        """
        # load vectorstore
        # with open(os.path.join(index_directory, 'vectorstore.pkl'), "rb") as f:
        #     vectorstore = pickle.load(f)
        log_debug(f"Loading index from: {index_directory}")
        embeddings = OpenAIEmbeddings(model=embedding_model, chunk_size=chunk_size)
        vectorstore = FAISS.load_local(index_directory, embeddings)
        return vectorstore

    def prepare_qa_chain(
            self, index_directory: str = None, temperature: float = 0.5,
            model_name: str = "gpt-3.5-turbo", max_tokens: int = 512
    ):
        """

        """
        log_debug(f"Preparing QA chain for: {index_directory}")

        # get document vectors
        faiss_index_path = os.path.join(index_directory, 'faiss')
        vectorstore = self.load_vectors(index_directory=faiss_index_path)

        # get q/a chain
        llm = ChatOpenAI(temperature=temperature, model_name=model_name, max_tokens=max_tokens)
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever())
        return qa_chain

    @staticmethod
    def ask_question(question: str = None, qa_chain: BaseConversationalRetrievalChain = None, chat_history: List[Tuple] = None):
        """

        """
        start = time.time()
        with get_openai_callback() as cb:
            result = qa_chain({"question": question, "chat_history": chat_history})
        total_time = round(time.time() - start, 2)
        answer = result["answer"]
        chat_history.append((question, answer))
        answer_meta = f"""Total tokens: {cb.total_tokens} (Prompt: {cb.prompt_tokens} + Completion: {cb.completion_tokens})
        Total requests: {cb.successful_requests}
        Total cost (USD): {round(cb.total_cost, 6)}"""
        return answer, answer_meta, chat_history
