from pdfminer.high_level import extract_text
from indexing_pipeline.vector_db import VectorDatabase
from indexing_pipeline.llm_handler import LLMHandler
from utils.scores import sort_jobs_by_score
from utils.utils import check_python_version, load_env_vars, suppress_warnings

if __name__ == "__main__":
    check_python_version()
    env_vars = load_env_vars(["INDEX_NAME", "PINECONE_API_KEY"])
    suppress_warnings()

    # Load the CV text
    text = extract_text("cvs/junior-full-stack-developer-resume-example.pdf")

    llm_handler = LLMHandler()
    applicant_profile = llm_handler.extract_data(text=text)
    print("Applicant profile:", applicant_profile)

    # Prepare query for job descriptions
    applicant_query = llm_handler.prepare_query(applicant_profile=applicant_profile)
    print("Applicant query:", applicant_query)

    # Initialize the Vector Database and search for job descriptions
    vector_db = VectorDatabase(index_name=env_vars["INDEX_NAME"], pinecone_api_key=env_vars["PINECONE_API_KEY"])
    search_results = vector_db.search(query=applicant_query, k=5)

    # Prepare inputs for comparison
    comparison_result = llm_handler.compare_applicant_with_jobs(applicant_profile=applicant_profile, job_descriptions_text=search_results)

    # Sort and display results
    sorted_jobs = sort_jobs_by_score(jobs=comparison_result)
    best_match = sorted_jobs[0]

    print(f"Recommended job source: {best_match['source']} with a final score of {best_match['final_score']:.2f}")
    for result in sorted_jobs:
        print(f"Source: {result['source']}, Final Score: {result['final_score']:.2f}")
