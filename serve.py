import time
import logging
import argparse
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import UploadFile, File, Depends
from typing import Any, Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware

from utils.rag_utils.agent_util import QueryAgent
from utils.rag_utils.rag_util import load_index_and_metadata
from utils.data_parsing_utils.document_parser import process_audio_files, process_document_files

app = FastAPI(
    title="Inference API for VerbalVista",
    description="🅛🅛🅜 + Your Data = 🩶",
    version="1.4",
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


class ProcessDataOutput(BaseModel):
    index_name: str
    meta: Dict[str, Any]


@serve.deployment()
@serve.ingress(app)
class VerbalVistaAssistantDeployment:
    """
    Initialize VerbalVista Ray Assistant class with index_directory!
    """

    def __init__(self, index_directory=None, logging_level=None):
        """
        Initialize the search agent with necessary indices and metadata.

        :param: index_directory (str, optional): The directory path where the indices and
            metadata are stored. If not provided, a default path or method will be used
            to load the required components.
        :param: logging_level: Logging level
        """
        self.logger = logging.getLogger("ray.serve")
        self.logger.setLevel(logging_level)

        # get faiss index, lexical index and metadata for given index directory
        index_meta = load_index_and_metadata(index_directory=index_directory)
        faiss_index = index_meta["faiss_index"]
        lexical_index = index_meta["lexical_index"]
        metadata_dict = index_meta["metadata_dict"]

        # get query agent
        self.query_agent = QueryAgent(
            faiss_index=faiss_index, metadata_dict=metadata_dict, lexical_index=lexical_index, reranker=None
        )

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
        result = self.query_agent(
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

    @staticmethod
    async def index_files(file: UploadFile = File(...), data: ProcessDataInput = Depends()) -> Dict[str, Any]:
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
        file_content = await file.read()

        # Construct a metadata dictionary from the processing data.
        meta = {
            "filename": file.filename,
            "chunk_size": data.chunk_size,
            "chunk_overlap": data.chunk_overlap,
            "embedding_model": data.embedding_model,
            "save_to_one_file": data.save_to_one_file
        }

        # Return a dictionary with the decoded file content as 'index_name' and the metadata.
        return {"index_name": file_content.decode("utf-8"), "meta": meta}

    @app.post("/process/documents")
    async def process_documents(
            self, file: UploadFile = File(...), data: ProcessDataInput = Depends()
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
    >> python serve.py --index_dir=../data/indices/my_index/
    >> ray stop
    """
    parser = argparse.ArgumentParser(description='Start VerbalVista Ray Server!')
    parser.add_argument('--index_dir', type=str, required=True, help='Index directory.')
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
    ).bind(index_directory=args.index_dir, logging_level=logging_level)

    serve.run(
        deployment,
        host=args.host,
        port=args.port,
        route_prefix=args.route_prefix,
        name=args.server_name
    )


if __name__ == "__main__":
    main()
