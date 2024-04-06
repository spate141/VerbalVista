import os
import pickle
from datetime import datetime
from itertools import zip_longest
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from utils import logger


class MetaData(BaseModel):
    llm_model: str = Field(...)
    embedding_model: str = Field(...)
    temperature: float = Field(...)
    tokens: Dict[str, int] = Field(...)
    cost: Dict[str, float] = Field(...)


class Message(BaseModel):
    role: str = Field(...)
    utc_timestamp: str = Field(...)
    content: Optional[str] = Field(...)
    meta: Optional[MetaData] = Field(default=None)


class ChatHistoryOutput(BaseModel):
    history: List[Message]

    @classmethod
    def parse_list(cls, obj_list: List[Dict[str, Any]]) -> 'ChatHistoryOutput':
        return cls(history=[Message(**obj) for obj in obj_list])


class ChatHistoryUtil:

    def __init__(self, chat_history_dir: str = None, index_name: str = None, server_logger=None):
        """
        Initializes the ChatHistoryUtil instance for managing chat histories, including loading and saving chat sessions.
        Chat sessions are stored as pickle files named after the index name within the specified chat history directory.

        :param chat_history_dir: Directory where chat histories are stored. Each chat history is saved in a subdirectory
                                 named after its index name.
        :param index_name: Unique identifier for the chat session. Used to name the subdirectory and pickle file.
        :param server_logger: Logger for recording chat session events. If not provided, a default logger is used.
        """
        self.index_name = index_name
        if server_logger:
            self.logger = server_logger
        else:
            self.logger = logger

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
        Appends a chat message and its metadata to the chat session and saves the updated chat history to disk.

        :param role: The role of the message sender (e.g., 'user', 'assistant'). Used for distinguishing between
                     different participants in the chat.
        :param content: The text content of the message.
        :param meta: Additional metadata associated with the message. This can include information such as the language
                     model used, generation parameters, and response metrics.
        """
        # Add messages to chat history
        chat_utc_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        self.chat_session[self.index_name]['messages'].append({"role": role, "content": content})
        self.chat_session[self.index_name]['timestamps'].append({"utc_time": chat_utc_time})
        self.chat_session[self.index_name]['meta'].append(meta)

        # Save conversation to local file
        with open(self.chat_history_filepath, 'wb') as f:
            pickle.dump(self.chat_session[self.index_name], f)

        self.logger.debug(f'Saved chat at: {chat_utc_time}')

    @staticmethod
    def sort_chat_history(chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts chat messages based on their timestamps to ensure the chat history is presented in chronological order.

        :param chat_history: A list of chat messages, where each message is represented by a dictionary containing
                             at least the UTC timestamp key.
        :return: The sorted list of chat messages in chronological order.
        """
        def pair_messages(iterable):
            args = [iter(iterable)] * 2
            return zip_longest(*args)

        paired_messages = list(pair_messages(chat_history))
        paired_messages.sort(key=lambda pair: pair[0]['utc_timestamp'], reverse=True)
        sorted_history = [message for pair in paired_messages for message in pair if message]
        return sorted_history

    def load_chat_history(self) -> List[Dict[str, Union[str, Any]]]:
        """
        Loads and returns the chat history for the current index name from the stored pickle file. The chat history is
        returned as a sorted list of messages, each with its role, content, timestamp, and optional metadata.

        :return: A list of dictionaries representing the chat history. Each dictionary includes the sender's role,
                 message content, timestamp, and, if available, metadata such as model details and generation parameters.
        """
        chat_history = []
        index_chat = self.chat_session[self.index_name]
        self.logger.debug(f'Loaded chat for index: {self.index_name}')
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
