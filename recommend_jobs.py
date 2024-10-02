import json
import os
import sys
import warnings

from pdfminer.high_level import extract_text
from dotenv import load_dotenv, find_dotenv
from indexing_pipeline.vector_db import VectorDatabase
from indexing_pipeline.llm_handler import LLMHandler
from metrics.scores import sort_jobs_by_score


if __name__ == "__main__":
    if sys.version_info > (3, 10, 99):
        raise Exception("This code is not compatible with version of Python higher than 3.10")

    load_dotenv(find_dotenv())
    index_name = os.environ["INDEX_NAME"]
    pinecone_api_key = os.environ["PINECONE_API_KEY"]

    if not pinecone_api_key or len(pinecone_api_key) == 0:
        raise Exception("Pinecone API key is missing")

    if not index_name or len(index_name) == 0:
        raise Exception("Index name is missing")

    # Ignore LanguageChain deprecated warnings
    warnings.filterwarnings("ignore")

    # Load the CV text
    text = extract_text("cvs/CV-DESARROLLADOR.pdf")

    # Initialize LLM Handler
    llm_handler = LLMHandler()

    applicant_profile = llm_handler.extract_data(text=text)
    print("Applicant profile:")
    print(applicant_profile)

    # Prepare query for job descriptions
    applicant_query = llm_handler.prepare_query(applicant_profile=applicant_profile)
    print("Applicant query:")
    print(applicant_profile)

    # Initialize the Vector Database and search for job descriptions
    vector_db = VectorDatabase(index_name=index_name, pinecone_api_key=pinecone_api_key)
    search_results = vector_db.search(query=applicant_query, k=5)
    print("Search results:")
    print(search_results)

    # Combine job descriptions for prompt
    job_descriptions_text = "\n\n".join([f"Source: {result.metadata['source']}\nText: {result.page_content}" for result in search_results])

    # Prepare inputs for comparison
    comparison_result = llm_handler.compare_applicant_with_jobs(applicant_profile=applicant_profile, job_descriptions_text=job_descriptions_text)

    # Sort and display results
    sorted_jobs = sort_jobs_by_score(jobs=comparison_result)
    best_match = sorted_jobs[0]

    print(f"Recommended job source: {best_match['source']} with a final score of {best_match['final_score']:.2f}")
    for result in sorted_jobs:
        print(f"Source: {result['source']}, Final Score: {result['final_score']:.2f}")
