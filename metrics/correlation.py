import matplotlib.pyplot as plt

from models.rag import RAG
from utils.utils import suppress_warnings, check_python_version


def evaluate_rag_performance(retrieved_sorted_jobs_source, expected_ordered_output):
    y_labels = list(set(expected_ordered_output) | set(retrieved_sorted_jobs_source))

    x_positions_expected = list(range(1, len(expected_ordered_output) + 1))
    y_positions_expected = list(range(1, len(expected_ordered_output) + 1))

    y_positions_retrieved = [y_labels.index(job) + 1 if job in y_labels else len(y_labels) for job in retrieved_sorted_jobs_source]
    x_positions_retrieved = list(range(1, len(retrieved_sorted_jobs_source) + 1))

    print(f"Orden esperado: {expected_ordered_output}")
    print(f"Orden devuelto por RAG: {retrieved_sorted_jobs_source}")

    fig, ax = plt.subplots()

    ax.plot(x_positions_expected, y_positions_expected, marker='o', color='blue', label='Expected')
    ax.plot(x_positions_retrieved, y_positions_retrieved, marker='x', color='orange', label='Retured by RAG')

    ax.set_xlabel('Ranking')
    ax.set_ylabel('Job')
    ax.set_yticks(list(range(1, len(y_labels) + 1)))
    ax.set_yticklabels(y_labels, fontsize=8)
    # ax.set_title('Comparation between expected and retrieved jobs')

    plt.grid()
    plt.xticks(range(1, max(len(retrieved_sorted_jobs_source), len(expected_ordered_output)) + 1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_python_version()
    suppress_warnings()

    best_match, sorted_jobs = RAG().run_with_scores(pdf_path="../cvs/junior-full-stack-developer-resume-example.pdf", k=5)
    retrieved_sorted_jobs_source = [job['source'].split('.')[0] for job in sorted_jobs]

    expected_ordered_output = ["full-stack-developer", "backend-developer", "game-developer"]
    evaluate_rag_performance(retrieved_sorted_jobs_source, expected_ordered_output)
