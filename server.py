import os
import time
import logging
import argparse
from ray import serve
from fastapi import FastAPI
from dotenv import load_dotenv; load_dotenv()
from fastapi import UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware

from utils.openai_utils import OpenAIWisperUtil
from utils.server_utils.query_util import QueryUtil, QuestionInput, QuestionOutput
from utils.server_utils.process_document_util import ProcessDocumentsUtil, ProcessDataInput, ProcessDataOutput


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
        start = time.time()
        query_util = QueryUtil(indices_dir=self.indices_dir, index_name=query.index_name)
        end = time.time()
        self.logger.info(f"Query agent initiated in {round((end - start) * 1000, 2)} ms")
        result = query_util.predict(
            query=query.query, temperature=query.temperature, embedding_model=query.embedding_model,
            llm_model=query.llm, max_semantic_retrieval_chunks=query.max_semantic_retrieval_chunks,
            max_lexical_retrieval_chunks=query.max_lexical_retrieval_chunks
        )
        end = time.time()
        self.logger.info(f"Finished /query in {round((end - start) * 1000, 2)} ms")
        return QuestionOutput.parse_obj(result)

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

        # Initialize Process Documents Util Class
        process_doc_util = ProcessDocumentsUtil(
            indices_dir=self.indices_dir, document_dir=self.document_dir, tmp_audio_dir=self.tmp_audio_dir,
            openai_wisper_util=self.openai_wisper_util
        )

        # (1) Read the file content
        start = time.time()
        file_meta = await process_doc_util.read_file(file, file_description=data.file_description)
        end = time.time()
        self.logger.info(f"Finished reading `{file_meta['name']}` in {round((end - start) * 1000, 2)} ms")

        # (2) Process file content and extract text
        start = time.time()
        extracted_texts = process_doc_util.extract_text(file_meta=file_meta)
        end = time.time()
        self.logger.info(f"Text Extracted from `{file_meta['name']}` in {round((end - start) * 1000, 2)} ms")

        # (3) Write extracted text to tmp file
        start = time.time()
        tmp_document_save_path = process_doc_util.save_extracted_text(
            extracted_texts=extracted_texts, single_file_flag=data.save_to_one_file
        )
        end = time.time()
        self.logger.info(f"Extracted text saved to `{tmp_document_save_path}` in {round((end - start) * 1000, 2)} ms")

        # (4) Generate FAISS index
        start = time.time()
        result = process_doc_util.generate_faiss_index(
            local_doc_filepath=tmp_document_save_path, chunk_size=data.chunk_size, chunk_overlap=data.chunk_overlap,
            embedding_model=data.embedding_model
        )
        end = time.time()
        _index_dir = result.pop("index_directory")
        self.logger.info(f"FAISS index saved to `{_index_dir}` in {round((end - start) * 1000, 2)} ms")

        # (5) Construct a metadata dictionary from the processing data.
        _ = file_meta.pop("file")
        result["meta"] = {
            "file_meta": file_meta,
            "chunk_size": data.chunk_size,
            "chunk_overlap": data.chunk_overlap,
            "embedding_model": data.embedding_model,
            "save_to_one_file": data.save_to_one_file
        }
        end = time.time()
        self.logger.info(f"Finished /process/documents in {round((end - start) * 1000, 2)} ms")
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
