from models.rag import RAG
from utils.utils import check_python_version, suppress_warnings


# https://medium.com/@ismailbenlemsieh/how-to-choose-the-right-metric-to-evaluate-your-classification-model-30e4569021db
def calculate_metrics(expected_output, search_results_source):
    true_positives = sum([1 for source in search_results_source if source in expected_output])
    false_positives = sum([1 for source in search_results_source if source not in expected_output])
    false_negatives = sum([1 for source in expected_output if source not in search_results_source])

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score


def print_metrics_table(precision, recall, f1_score):
    print("\n\n")
    print("+" + "-" * 50 + "+")
    print("|{:^50}|".format("Evaluation Metrics"))
    print("+" + "-" * 50 + "+")
    print("|{:<20}|{:<30}|".format("Metric", "Value"))
    print("+" + "-" * 50 + "+")
    print("|{:<20}|{:<30.2f}|".format("Precision", precision))
    print("|{:<20}|{:<30.2f}|".format("Recall", recall))
    print("|{:<20}|{:<30.2f}|".format("F1-Score", f1_score))
    print("+" + "-" * 50 + "+")

    # Brief descriptions of each metric
    print("\nDescription of Metrics:")
    print("- Precision: Measures the percentage of relevant documents among the retrieved ones.")
    print("- Recall: Measures the percentage of relevant documents that were successfully retrieved.")
    print("- F1-Score: The harmonic mean of Precision and Recall, balancing the two metrics.")


if __name__ == "__main__":
    check_python_version()
    suppress_warnings()

    _, _, search_results = RAG().run(pdf_path="../cvs/backend-developer.pdf", k=4)

    expected_output = ["game-developer.txt", "backend-developer.txt", "full-stack-developer.txt", "frontend-developer.txt"]
    search_results_source = [result.split("Source: ")[1].split("\n")[0] for result in search_results.split("\n\n") if "Source:" in result]

    precision, recall, f1_score = calculate_metrics(expected_output, search_results_source)
    print_metrics_table(precision, recall, f1_score)
