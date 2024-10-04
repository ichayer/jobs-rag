import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from models.rag import RAG
from utils.utils import suppress_warnings, check_python_version


def calculate_pearson(expected_positions, retrieved_positions):
    min_len = min(len(expected_positions), len(retrieved_positions))
    if min_len < 2:
        return None
    return pearsonr(expected_positions[:min_len], retrieved_positions[:min_len])

def plot_ranking_comparison(expected_indices, expected_positions, retrieved_indices, retrieved_positions, job_labels):
    fig, ax = plt.subplots()

    ax.plot(expected_indices, expected_positions, marker='o', color='blue', label='Expected')
    ax.plot(retrieved_indices, retrieved_positions, marker='x', color='orange', label='Returned by RAG')

    ax.set_xlabel('Ranking')
    ax.set_ylabel('Job')
    ax.set_yticks(list(range(1, len(job_labels) + 1)))
    ax.set_yticklabels(job_labels, fontsize=8)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.grid()
    plt.xticks(range(1, max(len(expected_indices), len(retrieved_indices)) + 1))
    plt.tight_layout()
    plt.show()

def compare_rag_results(retrieved_jobs, expected_jobs):
    unique_job_labels = sorted(list(set(expected_jobs) | set(retrieved_jobs)), key=lambda x: expected_jobs.index(x) if x in expected_jobs else float('inf'))

    expected_job_positions = [unique_job_labels.index(job) + 1 for job in expected_jobs]
    retrieved_job_positions = [unique_job_labels.index(job) + 1 for job in retrieved_jobs]

    expected_ranking_indices = list(range(1, len(expected_jobs) + 1))
    retrieved_ranking_indices = list(range(1, len(retrieved_jobs) + 1))

    return expected_ranking_indices, expected_job_positions, retrieved_ranking_indices, retrieved_job_positions, unique_job_labels


if __name__ == "__main__":
    check_python_version()
    suppress_warnings()

    best_match, sorted_jobs = RAG().run_with_scores(pdf_path="../cvs/junior-full-stack-developer-resume-example.pdf", k=5)
    retrieved_jobs = [job['source'].split('.')[0] for job in sorted_jobs]

    expected_jobs = ["full-stack-developer", "backend-developer", "game-developer"]
    expected_ranking_indices, expected_job_positions, retrieved_ranking_indices, retrieved_job_positions, unique_job_labels = \
        compare_rag_results(retrieved_jobs, expected_jobs)

    pearson_corr = calculate_pearson(expected_job_positions, retrieved_job_positions)

    # Results
    print("\n\n")
    print("+" + "-" * 50 + "+")
    print("|{:^50}|".format("Results"))
    print("+" + "-" * 50 + "+")

    print("|{:^50}|".format("Ranking Comparison"))
    print("+" + "-" * 50 + "+")
    print("|{:<25}| {:<22}|".format("Expected Order", "RAG Retrieved Order"))
    for expected, retrieved in zip(expected_jobs, retrieved_jobs):
        print("|{:<25}| {:<22}|".format(expected, retrieved))

    print("+" + "-" * 50 + "+")
    print("|{:^50}|".format("Position Comparison"))
    print("+" + "-" * 50 + "+")
    print("|{:<25}| {:<22}|".format("Expected Positions", "RAG Positions"))
    for expected_pos, retrieved_pos in zip(expected_job_positions, retrieved_job_positions):
        print("|{:<25}| {:<22}|".format(expected_pos, retrieved_pos))

    print("+" + "-" * 50 + "+")
    if pearson_corr is None:
        print("|{:<50}|".format("Not enough data to calculate Pearson correlation."))
    else:
        print("|{:<50}|".format(f"Pearson correlation: {pearson_corr[0]:.2f}"))
    print("+" + "-" * 50 + "+")
    print("\n\n")

    plot_ranking_comparison(expected_ranking_indices, expected_job_positions, retrieved_ranking_indices, retrieved_job_positions, unique_job_labels)
