from pdfminer.high_level import extract_text
from indexing_pipeline.llm_handler import LLMHandler
from indexing_pipeline.vector_db import VectorDatabase
from utils.utils import check_python_version, load_env_vars, suppress_warnings

if __name__ == "__main__":
    check_python_version()
    env_vars = load_env_vars(["INDEX_NAME", "PINECONE_API_KEY"])
    suppress_warnings()

    text = extract_text("../cvs/junior-full-stack-developer-resume-example.pdf")

    llm_handler = LLMHandler()
    applicant_profile = llm_handler.extract_data(text=text)
    applicant_query = llm_handler.prepare_query(applicant_profile=applicant_profile)

    vector_db = VectorDatabase(index_name=env_vars["INDEX_NAME"], pinecone_api_key=env_vars["PINECONE_API_KEY"])
    search_results = vector_db.search(query=applicant_query, k=5)

    expected_output = ["full-stack-developer.txt"]
    search_results_source = [result.split("Source: ")[1].split("\n")[0] for result in search_results.split("\n\n") if "Source:" in result]

    true_positive_sum = sum([1 for source in search_results_source if source in expected_output])
    false_positive_sum = sum([1 for source in search_results_source if source not in expected_output])
    precision = true_positive_sum / (true_positive_sum + false_positive_sum)

    print(f"Retrieved documents: {search_results_source}")
    print(f"Expected documents: {expected_output}")
    print(f"Precision: {precision:.2f}")
