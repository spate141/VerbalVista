import os
import time
import logging
import argparse
from ray import serve
from io import BytesIO
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional
from dotenv import load_dotenv; load_dotenv()
from fastapi import UploadFile, File, Depends, Form
from fastapi.middleware.cors import CORSMiddleware

from utils.openai_utils import OpenAIWisperUtil
from utils.rag_utils.agent_util import QueryAgent
from utils.data_parsing_utils import write_data_to_file
from utils.rag_utils.rag_util import load_index_and_metadata, index_data
from utils.data_parsing_utils.document_parser import process_audio_files, process_document_files

app = FastAPI(
    title="Inference API for VerbalVista",
    description="🅛🅛🅜 + Your Data = 🩶",
    version="1.5",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionInput(BaseModel):
    query: str
    index_name: str
    llm: Optional[str] = "gpt-3.5-turbo"
    embedding_model: Optional[str] = "text-embedding-ada-002"
    temperature: Optional[float] = 0.5
    max_semantic_retrieval_chunks: Optional[int] = 5
    max_lexical_retrieval_chunks: Optional[int] = 1


class QuestionOutput(BaseModel):
    query: str
    answer: str
    completion_meta: Dict[str, Any]


class ProcessDataInput(BaseModel):
    chunk_size: Optional[int] = 600
    chunk_overlap: Optional[int] = 30
    embedding_model: Optional[str] = "text-embedding-ada-002"
    save_to_one_file: Optional[bool] = False
    filemeta: Optional[str] = ""

    @classmethod
    def as_form(
        cls,
        chunk_size: int = Form(600),
        chunk_overlap: int = Form(30),
        embedding_model: str = Form("text-embedding-ada-002"),
        save_to_one_file: bool = Form(False),
        filemeta: str = Form("")
    ):
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            save_to_one_file=save_to_one_file,
            filemeta=filemeta
        )


class ProcessDataOutput(BaseModel):
    index_name: str
    meta: Dict[str, Any]


@serve.deployment()
@serve.ingress(app)
class VerbalVistaAssistantDeployment:
    """
    Initialize VerbalVista Ray Assistant class with index_directory!
    """
    def __init__(self, logging_level=None):
        """
        Initialize the search agent with necessary indices and metadata.
        :param: logging_level: Logging level
        """
        self.openai_wisper_util = OpenAIWisperUtil(api_key=os.getenv("OPENAI_API_KEY"))
        self.tmp_audio_dir = 'data/tmp_audio_dir/'
        self.document_dir = 'data/documents/'
        self.indices_dir = 'data/indices/'
        self.logger = logging.getLogger("ray.serve")
        self.logger.setLevel(logging_level)

    def predict(self, query: QuestionInput, stream: bool = False) -> Dict[str, Any]:
        """
        Generate predictions based on the input question.

        This method uses the internal query agent to process a given question and
        retrieve relevant information. It supports both semantic and lexical search
        mechanisms.

        :param: query (QuestionInput): An object containing the query parameters such as
          the actual query string, temperature setting for response variability,
          embedding model name for semantic search, LLM model name, and the number
          of chunks for semantic and lexical retrieval.
        :param: stream (bool, optional): A flag indicating whether to stream results
          continuously or not. Defaults to False.
        :return: Dict[str, Any]: The result from the query agent, typically including
          information relevant to the input query.
        """
        # Get faiss index, lexical index and metadata for given index directory
        start = time.time()
        index_meta = load_index_and_metadata(
            index_directory=os.path.join(self.indices_dir, query.index_name)
        )
        faiss_index = index_meta["faiss_index"]
        metadata_dict = index_meta["metadata_dict"]
        lexical_index = index_meta["lexical_index"]

        # Get query agent
        query_agent = QueryAgent(
            faiss_index=faiss_index,
            metadata_dict=metadata_dict,
            lexical_index=lexical_index,
            reranker=None
        )
        end = time.time()
        self.logger.info(f"Query Agent initiated in {round((end - start) * 1000, 2)} ms")

        # Generate prediction response
        result = query_agent(
            query=query.query,
            temperature=query.temperature,
            embedding_model_name=query.embedding_model,
            stream=stream,
            llm_model=query.llm,
            num_chunks=query.max_semantic_retrieval_chunks,
            lexical_search_k=query.max_lexical_retrieval_chunks,
        )
        return result

    @app.post("/query")
    def query(self, query: QuestionInput) -> QuestionOutput:
        """
        Handle POST request to '/query' endpoint.

        This method receives a query in the form of a QuestionInput object, processes it to predict an answer,
        and returns a QuestionOutput object containing the prediction result.

        :param: self: Refers to the instance of the class where this method is defined.
        :param: query: A QuestionInput object containing the query data.
        :return: QuestionOutput: An instance containing the predicted answer, parsed from the result object.
        """
        # Process the query to predict the answer.
        start = time.time()
        result = self.predict(query)
        end = time.time()
        self.logger.info(f"Finished /query in {round((end - start) * 1000, 2)} ms")

        # Parse the prediction result into a QuestionOutput object and return it.
        return QuestionOutput.parse_obj(result)

    async def index_files(
            self, file: UploadFile = File(...), data: ProcessDataInput = Depends(ProcessDataInput.as_form)
    ) -> Dict[str, Any]:
        """
        Asynchronously indexes files and returns metadata along with the index name.

        This static method reads the contents of the uploaded file and constructs a metadata dictionary
        based on the processing data provided. It decodes the file content into a UTF-8 string which is
        used as the index name.

        Parameters:
        - file: An UploadFile object representing the file to be indexed.
        - data: A ProcessDataInput object containing parameters for indexing such as chunk size,
                chunk overlap, embedding model, and whether to save to one file.

        Returns:
        - Dict[str, Any]: A dictionary with two keys: 'index_name', containing the decoded file content,
                           and 'meta', containing the metadata information.
        """

        # Read the file content asynchronously.
        start = time.time()
        file_content = await file.read()
        file_name = file.filename
        file_meta = {
            'name': file_name, 'type': file.content_type,
            'size': len(file_content), 'file': BytesIO(file_content)
        }

        # Extract text from input file
        full_documents = []
        extracted_text = ""
        if file_name.endswith(('.m4a', '.mp3', '.wav', '.webm', '.mp4', '.mpga', '.mpeg')):
            extracted_text = process_audio_files(
                tmp_audio_dir=self.tmp_audio_dir, file_meta=file_meta, openai_wisper_util=self.openai_wisper_util
            )
        elif file_name.endswith(('.pdf', '.docx', '.txt', '.eml')):
            extracted_text = process_document_files(file_meta=file_meta)
        full_documents.append({
            "file_name": file_name,
            "extracted_text": extracted_text,
            "doc_description": data.filemeta
        })
        end = time.time()
        self.logger.info(f"Text Extracted from `{file_name}` in {round((end - start) * 1000, 2)} ms")

        # Write extracted text to tmp file
        start = time.time()
        tmp_document_save_path = write_data_to_file(
            document_dir=self.document_dir,
            full_documents=full_documents,
            single_file_flag=data.save_to_one_file,
        )
        end = time.time()
        self.logger.info(f"Extracted text saved to `{tmp_document_save_path}` in {round((end - start) * 1000, 2)} ms")

        # FAISS index created from extracted text
        start = time.time()
        doc_dir = os.path.join(self.document_dir, tmp_document_save_path)
        index_dir = os.path.join(self.indices_dir, tmp_document_save_path)
        index_data(
            document_directory=doc_dir,
            index_directory=index_dir,
            chunk_size=data.chunk_size,
            embedding_model=data.embedding_model
        )
        end = time.time()
        self.logger.info(f"FAISS index saved to `{index_dir}` in {round((end - start) * 1000, 2)} ms")

        # Construct a metadata dictionary from the processing data.
        _ = file_meta.pop("file")
        meta = {
            "file_meta": file_meta,
            "chunk_size": data.chunk_size,
            "chunk_overlap": data.chunk_overlap,
            "embedding_model": data.embedding_model,
            "save_to_one_file": data.save_to_one_file
        }

        # Return a dictionary with the decoded file content as 'index_name' and the metadata.
        return {"index_name": os.path.basename(index_dir), "meta": meta}

    @app.post("/process/documents")
    async def process_documents(
            self, file: UploadFile = File(...), data: ProcessDataInput = Depends(ProcessDataInput.as_form)
    ) -> ProcessDataOutput:
        """
        Handle POST request to '/process/documents' endpoint.

        This asynchronous function processes documents by indexing them and logging the operation.
        It accepts a file and additional processing data, then returns processed data output.

        :param: self: Refers to the instance of the class where this method is defined.
        :param: file: An UploadedFile object that contains the file to be processed.
        :param: data: A ProcessDataInput object containing additional data for processing.
        :return: ProcessDataOutput: An instance containing the processed result, parsed from the result object.
        """
        # Index the provided file with the associated data asynchronously.
        start = time.time()
        result = await self.index_files(file, data)
        end = time.time()

        self.logger.info(f"Finished /process/documents in {round((end - start) * 1000, 2)} ms")

        # Parse the result into a ProcessDataOutput object and return it.
        return ProcessDataOutput.parse_obj(result)


def main():
    """
    How to start and stop server?
    >> ray start --head
    >> python server.py --index_dir=../data/indices/my_index/
    >> ray stop
    """
    parser = argparse.ArgumentParser(description='Start VerbalVista Ray Server!')
    parser.add_argument('--num_replicas', type=int, default=1, help='Number of replicas.')
    parser.add_argument('--num_cpus', type=int, default=4, help='Number of CPUs.')
    parser.add_argument('--num_gpus', type=int, default=0, help='Number of GPUs.')
    parser.add_argument('--max_concurrent_queries', type=int, default=100, help='Max concurrent queries.')
    parser.add_argument('--health_check_period_s', type=int, default=10, help='Health check period (seconds).')
    parser.add_argument('--health_check_timeout_s', type=int, default=30, help='Health check timeout (seconds).')
    parser.add_argument('--graceful_shutdown_timeout_s', type=int, default=20, help='Graceful shutdown timeout (seconds).')
    parser.add_argument('--graceful_shutdown_wait_loop_s', type=int, default=2, help='Graceful shutdown wait loop (seconds).')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address.')
    parser.add_argument('--port', type=int, default=8000, help='Port number.')
    parser.add_argument('--route_prefix', type=str, default='/', help='Route prefix.')
    parser.add_argument('--server_name', type=str, default='verbal_vista', help='Server name.')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (e.g., -v, -vv)')

    args = parser.parse_args()

    # Set logging level based on verbosity count
    if args.verbose >= 2:
        logging_level = logging.DEBUG
    elif args.verbose == 1:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARNING

    deployment = VerbalVistaAssistantDeployment.options(
        name="VerbalVistaServer",
        num_replicas=args.num_replicas,
        ray_actor_options={"num_cpus": args.num_cpus, "num_gpus": args.num_gpus},
        max_concurrent_queries=args.max_concurrent_queries,
        health_check_period_s=args.health_check_period_s,
        health_check_timeout_s=args.health_check_timeout_s,
        graceful_shutdown_timeout_s=args.graceful_shutdown_timeout_s,
        graceful_shutdown_wait_loop_s=args.graceful_shutdown_wait_loop_s
    ).bind(logging_level=logging_level)

    serve.run(
        deployment,
        host=args.host,
        port=args.port,
        route_prefix=args.route_prefix,
        name=args.server_name
    )


if __name__ == "__main__":
    main()
