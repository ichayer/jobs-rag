import os
import sys

from dotenv import load_dotenv, find_dotenv

from indexing_pipeline.document_loader import DocumentLoader
from indexing_pipeline.vector_db import VectorDatabase

if __name__ == "__main__":
    if sys.version_info[:2] != (3, 10):
        raise Exception("This code is only compatible with Python 3.10")

    load_dotenv(find_dotenv())

    index_name = os.environ["INDEX_NAME"]
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    jobs_folder_path = os.environ["JOBS_FOLDER"]

    if not pinecone_api_key or len(pinecone_api_key) == 0:
        raise Exception("Pinecone API key is missing")

    if not index_name or len(index_name) == 0:
        raise Exception("Index name is missing")

    if not jobs_folder_path or len(jobs_folder_path) == 0:
        raise Exception("Jobs folder is missing")

    document_loader = DocumentLoader(jobs_folder_path=jobs_folder_path)
    vector_store = VectorDatabase(index_name=index_name, pinecone_api_key=pinecone_api_key)

    job_descriptions = document_loader.load_job_descriptions()
    vector_store.add_documents(job_descriptions)

    print("Done!")
