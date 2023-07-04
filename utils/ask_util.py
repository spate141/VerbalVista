import os
import time
import pickle
from typing import Dict, List, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain


class AskUtil:

    def __init__(self):
        pass

    @staticmethod
    def load_vectors(index_directory: str = None):
        """

        """
        # load vectorstore
        with open(os.path.join(index_directory, 'vectorstore.pkl'), "rb") as f:
            vectorstore = pickle.load(f)
        return vectorstore

    def prepare_qa_chain(self, index_directory: str = None, temperature: float = 0.5, model_name: str = "gpt-3.5-turbo", max_tokens: int = 512):
        """

        """
        # get document vectors
        vectorstore = self.load_vectors(index_directory=index_directory)

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
        answer_meta = f"""
        ```markdown
        - Total Tokens: {cb.total_tokens}
            - Prompt: {cb.prompt_tokens} + Response: {cb.completion_tokens}
        - Total Cost (USD): ${round(cb.total_cost, 4)}
        - Total Time (Seconds): {total_time}
        ```
        """
        answer = result["answer"]
        chat_history.append((question, answer))
        return answer, answer_meta, chat_history
