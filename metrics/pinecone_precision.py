import os
import sys
import warnings

from pdfminer.high_level import extract_text
from dotenv import find_dotenv, load_dotenv

from indexing_pipeline.llm_handler import LLMHandler
from indexing_pipeline.vector_db import VectorDatabase


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 10):
        raise Exception("This code is only compatible with Python 3.10")

    load_dotenv(find_dotenv())
    index_name = os.environ["INDEX_NAME"]
    pinecone_api_key = os.environ["PINECONE_API_KEY"]

    if not pinecone_api_key or len(pinecone_api_key) == 0:
        raise Exception("Pinecone API key is missing")

    if not index_name or len(index_name) == 0:
        raise Exception("Index name is missing")

    warnings.filterwarnings("ignore")

    text = extract_text("../cvs/junior-full-stack-developer-resume-example.pdf")

    llm_handler = LLMHandler()
    applicant_profile = llm_handler.extract_data(text=text)
    applicant_query = llm_handler.prepare_query(applicant_profile=applicant_profile)

    vector_db = VectorDatabase(index_name=index_name, pinecone_api_key=pinecone_api_key)
    search_results = vector_db.search(query=applicant_query, k=5)

    expected_output = ["full-stack-developer.txt"]
    search_results_source = [result.split("Source: ")[1].split("\n")[0] for result in search_results.split("\n\n") if "Source:" in result]

    true_positive_sum = sum([1 for source in search_results_source if source in expected_output])
    false_positive_sum = sum([1 for source in search_results_source if source not in expected_output])

    precision = true_positive_sum / (true_positive_sum + false_positive_sum)

    print(f"Retrieved documents: {search_results_source}")
    print(f"Expected documents: {expected_output}")
    print(f"Precision: {precision:.2f}")
