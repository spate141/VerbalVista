import os
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict, Any
from utils.rag_utils.rag_util import get_available_indices


class IndexOutput(BaseModel):
    index_name: str
    index_description: str
    index_build_time: datetime

    class Config:
        arbitrary_types_allowed = True


class ListIndicesOutput(BaseModel):
    indices: List[IndexOutput]

    @classmethod
    def parse_list(cls, obj_list: List[Dict[str, Any]]) -> 'ListIndicesOutput':
        return cls(indices=[IndexOutput(**obj) for obj in obj_list])


class ListIndicesUtil:

    def __init__(self):
        """
        Initializes the ListIndicesUtil class without any parameters.
        """
        pass

    @staticmethod
    def list_indices(indices_dir: str = None):
        """
        Retrieves a list of index information from a specified directory.
        :param indices_dir: The directory path where index files are located. If None, a default path is used.
                            Defaults to None.
        :return: A list of dictionaries where each dictionary contains details about an index such as its name,
                 description, and build time.
        """
        index_df = get_available_indices(indices_dir=indices_dir)
        index_df.rename(columns={'Index Path': 'index_name', 'Index Description': 'index_description',
                                 'Creation Date': 'index_build_time'}, inplace=True)
        index_df['index_name'] = index_df['index_name'].apply(lambda x: os.path.basename(x))
        result = index_df.to_dict(orient='records')
        return result

