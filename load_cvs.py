import os
from pinecone_helper import get_vector_store
from langchain.schema import Document
from dotenv import load_dotenv, find_dotenv

def load_job_descriptions(folder_path):
    job_descriptions = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Assuming job descriptions are in text files
            with open(os.path.join(folder_path, filename), "r") as file:
                content = file.read()
                # Wrap content in a Document object and add metadata
                doc = Document(page_content=content, metadata={"source": filename})
                job_descriptions.append(doc)
    return job_descriptions

_ = load_dotenv(find_dotenv())

vectorstore = get_vector_store(os.environ['INDEX_NAME'])

# Load the job descriptions from the folder
job_descriptions = load_job_descriptions(os.environ['JOBS_FOLDER'])
vectorstore.add_documents(job_descriptions)