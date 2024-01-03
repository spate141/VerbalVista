import os
import pickle


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
        self.chat_session_state = {}

        # check if chat history is available locally, if yes; load the chat history
        if os.path.exists(self.chat_history_filepath):
            with open(self.chat_history_filepath, 'rb') as f:
                self.chat_session_state[self.index_name] = pickle.load(f)
        else:
            self.chat_session_state[self.index_name] = {'messages': [], 'meta': []}

    def save_chat(self, role: str = None, content: str = None, meta=None):
        """
        Saves a single chat message along with its metadata to the chat history.

        :param role: The role of the entity sending the message (e.g., 'user', 'assistant')
        :param content: The content of the message.
        :param meta: Additional metadata associated with the message.
        """
        # Add messages to chat history
        self.chat_session_state[self.index_name]['messages'].append({"role": role, "content": content})
        self.chat_session_state[self.index_name]['meta'].append(meta)

        # Save conversation to local file
        with open(self.chat_history_filepath, 'wb') as f:
            pickle.dump(self.chat_session_state[self.index_name], f)
