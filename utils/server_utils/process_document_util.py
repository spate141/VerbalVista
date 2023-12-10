import os
from io import BytesIO
from fastapi import Form
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from utils.rag_utils.rag_util import index_data
from utils.data_parsing_utils import write_data_to_file
from utils.data_parsing_utils.document_parser import process_audio_files, process_document_files


class ProcessDataInput(BaseModel):
    chunk_size: Optional[int] = 600
    chunk_overlap: Optional[int] = 30
    embedding_model: Optional[str] = "text-embedding-ada-002"
    save_to_one_file: Optional[bool] = False
    file_description: Optional[str] = ""

    @classmethod
    def as_form(
        cls,
        chunk_size: int = Form(600),
        chunk_overlap: int = Form(30),
        embedding_model: str = Form("text-embedding-ada-002"),
        save_to_one_file: bool = Form(False),
        file_description: str = Form("")
    ):
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            save_to_one_file=save_to_one_file,
            file_description=file_description
        )


class ProcessDataOutput(BaseModel):
    index_name: str
    index_meta: Dict[str, Any]


class ProcessDocumentsUtil:

    def __init__(
            self, indices_dir: str = None, document_dir: str = None, tmp_audio_dir: str = None, openai_wisper_util=None
    ):
        self.indices_dir = indices_dir
        self.document_dir = document_dir
        self.tmp_audio_dir = tmp_audio_dir
        self.openai_wisper_util = openai_wisper_util

    @staticmethod
    async def read_file(file, file_description: str = None) -> Dict[str, Any]:
        """
        Read the content of the file.
        :param: file: FastAPI File object
        :param: file_description: File description defined by user in API endpoint
        :return: Dict[str, Any]: File metadata
        """
        file_content = await file.read()
        file_meta = {
            'file': BytesIO(file_content),
            'name': file.filename,
            'type': file.content_type,
            'size': len(file_content),
            'description': file_description
        }
        return file_meta

    def extract_text(self, file_meta: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Extract text from the previously read file object based on file type.
        :param: file_meta: Dict[str, Any]: File metadata
        :return: List[Dict[str, Any]]: List of extracted text object dictionary.
        """
        extracted_texts = []
        extracted_text = ""
        file_name = file_meta['name']
        file_desc = file_meta['description']
        if file_name.endswith(('.m4a', '.mp3', '.wav', '.webm', '.mp4', '.mpga', '.mpeg')):
            extracted_text = process_audio_files(
                tmp_audio_dir=self.tmp_audio_dir, file_meta=file_meta, openai_wisper_util=self.openai_wisper_util
            )
        elif file_name.endswith(('.pdf', '.docx', '.txt', '.eml')):
            extracted_text = process_document_files(file_meta=file_meta)
        extracted_texts.append({
            "file_name": file_name,
            "extracted_text": extracted_text,
            "doc_description": file_desc
        })
        return extracted_texts

    def save_extracted_text(self, extracted_texts: List[Dict[str, Any]] = None, single_file_flag: bool = False) -> str:
        """
        Save the previously extracted text objects to tmp local file.
        :param: extracted_texts: List of extracted text object dictionary
        :param: single_file_flag: Boolean flag to indicate if to save all extracted texts objects into single file or not
        :return: file path of saved data.
        """
        tmp_document_save_path = write_data_to_file(
            document_dir=self.document_dir,
            full_documents=extracted_texts,
            single_file_flag=single_file_flag,
        )
        return tmp_document_save_path

    def generate_faiss_index(
            self, local_doc_filepath: str = None, chunk_size: int = None, chunk_overlap: int = None,
            embedding_model: str = None
    ) -> Dict[str, Any]:
        """
        Generate FAISS index from previously extracted and saved file object.
        :param: local_doc_filepath:
        :param: chunk_size:
        :param: chunk_overlap:
        :param: embedding_model:
        :return: Index dict metadata.
        """
        # FAISS index created from extracted text
        document_directory = os.path.join(self.document_dir, local_doc_filepath)
        index_directory = os.path.join(self.indices_dir, local_doc_filepath)
        index_data(
            document_directory=document_directory,
            index_directory=index_directory,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model
        )

        # Return a dictionary with the decoded file content as 'index_name' and the metadata.
        return {"index_name": os.path.basename(index_directory), "index_directory": index_directory}

