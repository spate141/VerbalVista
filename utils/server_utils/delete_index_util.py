import os
from pydantic import BaseModel
from utils.rag_utils.rag_util import delete_directory


class DeleteIndexOutput(BaseModel):
    index_name: str
    status: str


class DeleteIndexUtil:
    """
    Utility class for deleting an index.

    Methods:
        delete_index(index_dir: str = None, index_name: str = None) -> DeleteIndexOutput:
            Deletes the directory associated with a given index and returns the status.

            :param index_dir: The base directory where the index is located. Optional.
            :param index_name: The name of the index to be deleted. Optional.
            :return: An instance of DeleteIndexOutput containing the index_name and deletion status.
    """
    def __init__(self):
        pass

    @staticmethod
    def delete_index(index_dir: str = None, index_name: str = None):
        """
        Static method to delete the directory associated with a given index.

        :param index_dir: The base directory where the index is located, defaults to None.
        :param index_name: The name of the index to be deleted, defaults to None.
        :return: A dictionary with 'index_name' set to the name of the index and 'status' indicating
                 whether the deletion was successful ('removed') or not ('failed').
        """
        status = delete_directory(selected_directory=os.path.join(index_dir, index_name))
        return {"index_name": index_name, "status": "removed" if status else "failed"}

