import os
from pydantic import BaseModel
from utils.rag_utils.rag_util import delete_directory


class DeleteIndexOutput(BaseModel):
    index_name = str
    status: str


class DeleteIndexUtil:

    def __init__(self):
        pass

    @staticmethod
    def delete_index(index_dir: str = None, index_name: str = None):
        status = delete_directory(selected_directory=os.path.join(index_dir, index_name))
        return {"index_name": index_name, "status": "removed" if status else "failed"}

