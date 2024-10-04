from indexing_pipeline.document_loader import DocumentLoader
from indexing_pipeline.vector_db import VectorDatabase
from utils.utils import load_env_vars, check_python_version, suppress_warnings

if __name__ == "__main__":
    check_python_version()
    env_vars = load_env_vars(["INDEX_NAME", "PINECONE_API_KEY", "JOBS_FOLDER", "COMPUTE_DEVICE"])
    suppress_warnings()

    document_loader = DocumentLoader(jobs_folder_path=env_vars["JOBS_FOLDER"])
    vector_store = VectorDatabase(index_name=env_vars["INDEX_NAME"], pinecone_api_key=env_vars["PINECONE_API_KEY"])

    job_descriptions = document_loader.load_job_descriptions()
    vector_store.add_documents(job_descriptions)

    print("Done!")
