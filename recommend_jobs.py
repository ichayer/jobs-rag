import sys
from models.rag import RAG
from utils.utils import check_python_version, suppress_warnings

if __name__ == "__main__":
    check_python_version()
    suppress_warnings()

    if len(sys.argv) != 2:
        print("You must specify one parameter with the name of the CV file")
        exit(1)

    best_match, sorted_jobs = RAG().run_with_scores(pdf_path=sys.argv[1], k=5)
    print(f"Recommended job source: {best_match['source']} with a final score of {best_match['final_score']:.2f}")
    for result in sorted_jobs:
        print(f"Source: {result['source']}, Final Score: {result['final_score']:.2f}")
