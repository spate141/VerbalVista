import os
import pickle
from datetime import datetime
from itertools import zip_longest
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class MetaData(BaseModel):
    llm_model: str = Field(...)
    embedding_model: str = Field(...)
    temperature: float = Field(...)
    tokens: Dict[str, int] = Field(...)
    cost: Dict[str, float] = Field(...)


class Message(BaseModel):
    role: str = Field(...)
    utc_timestamp: str = Field(...)
    content: str = Field(...)
    meta: Optional[MetaData] = Field(default=None)


class ChatHistoryOutput(BaseModel):
    history: List[Message]

    @classmethod
    def parse_list(cls, obj_list: List[Dict[str, Any]]) -> 'ChatHistoryOutput':
        return cls(history=[Message(**obj) for obj in obj_list])


class ChatHistoryUtil:

    def __init__(self, chat_history_dir: str = None, index_name: str = None):
        """
        Initializes the ChatHistoryUtil object with a directory for storing chat history and
        an index name to identify it.
        :param chat_history_dir: The base directory where chat histories are stored. If None, no directory is set.
        :param index_name: A unique identifier for the chat history. If None, no index name is set.
        """
        self.index_name = index_name

        # Initialize chat history saving mechanism
        chat_dir_path = os.path.join(chat_history_dir, self.index_name)
        if not os.path.exists(chat_dir_path):
            os.makedirs(chat_dir_path)
        self.chat_history_filepath = os.path.join(chat_dir_path, f"{self.index_name}.pickle")

        # Initialize chat history
        self.chat_session = {}

        # check if chat history is available locally, if yes; load the chat history
        if os.path.exists(self.chat_history_filepath):
            with open(self.chat_history_filepath, 'rb') as f:
                self.chat_session[self.index_name] = pickle.load(f)
        else:
            self.chat_session[self.index_name] = {'messages': [], 'timestamps': [], 'meta': []}

    def save_chat(self, role: str = None, content: str = None, meta=None):
        """
        Saves a single chat message along with its metadata to the chat history.

        :param role: The role of the entity sending the message (e.g., 'user', 'assistant')
        :param content: The content of the message.
        :param meta: Additional metadata associated with the message.
        """
        # Add messages to chat history
        self.chat_session[self.index_name]['messages'].append({"role": role, "content": content})
        self.chat_session[self.index_name]['timestamps'].append({"utc_time": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')})
        self.chat_session[self.index_name]['meta'].append(meta)

        # Save conversation to local file
        with open(self.chat_history_filepath, 'wb') as f:
            pickle.dump(self.chat_session[self.index_name], f)

    @staticmethod
    def sort_chat_history(chat_history):
        """
        Sorts a list of chat messages in descending order based on their UTC timestamps.

        :param chat_history: List of dictionaries where each dictionary represents a chat message and contains a key 'utc_timestamp'.
        :return: A sorted list of chat messages in descending order by 'utc_timestamp'.
        """
        def pair_messages(iterable):
            args = [iter(iterable)] * 2
            return zip_longest(*args)

        paired_messages = list(pair_messages(chat_history))
        paired_messages.sort(key=lambda pair: pair[0]['utc_timestamp'], reverse=True)
        sorted_history = [message for pair in paired_messages for message in pair if message]
        return sorted_history

    def load_chat_history(self):
        """
        Retrieves and formats the chat history from a chat session indexed by 'index_name'.

        :return: A list of dictionaries, each representing a message with its associated metadata.
        Each dictionary contains the sender's role, message content, and optional metadata such as
        model used, temperature for generation, tokens, and cost if available.
        """
        chat_history = []
        index_chat = self.chat_session[self.index_name]
        for message, meta, timestamp in zip(index_chat['messages'], index_chat['meta'], index_chat['timestamps']):
            if meta:
                chat_history.append({
                    "role": message['role'],
                    "utc_timestamp": timestamp['utc_time'],
                    "content": message['content'],
                    "meta": {
                        "llm_model": meta['llm_model'],
                        "embedding_model": meta['embedding_model'],
                        "temperature": meta['temperature'],
                        "tokens": meta['tokens'],
                        "cost": meta['cost']
                    }
                })
            else:
                chat_history.append({
                    "role": message['role'],
                    "utc_timestamp": timestamp['utc_time'],
                    "content": message['content'],
                    "meta": None
                })
        sorted_chat_history = self.sort_chat_history(chat_history)
        return sorted_chat_history
