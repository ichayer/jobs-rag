from models.rag import RAG
from utils.utils import check_python_version, suppress_warnings

if __name__ == "__main__":
    check_python_version()
    suppress_warnings()

    _, _, search_results = RAG().run(pdf_path="../cvs/junior-full-stack-developer-resume-example.pdf", k=5)

    expected_output = ["full-stack-developer.txt"]
    search_results_source = [result.split("Source: ")[1].split("\n")[0] for result in search_results.split("\n\n") if "Source:" in result]

    true_positive_sum = sum([1 for source in search_results_source if source in expected_output])
    false_positive_sum = sum([1 for source in search_results_source if source not in expected_output])
    precision = true_positive_sum / (true_positive_sum + false_positive_sum)

    print(f"Retrieved documents: {search_results_source}")
    print(f"Expected documents: {expected_output}")
    print(f"Precision: {precision:.2f}")
