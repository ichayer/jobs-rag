
# Weights for each criterion
_weights = {
    "matching_skills": 0.4,  # 40% weight
    "missing_skills": -0.1,  # Penalize missing skills
    "relevant_jobs": 0.25,   # 25% weight
    "relevant_degrees": 0.1, # 10% weight
    "location_match": 0.15   # 15% weight
}

def _calculate_job_score(job):
    print(job)
    # Calculate the final score for each job description
    score = (job['matching_skills'] * _weights['matching_skills'] +
             job['missing_skills'] * _weights['missing_skills'] +
             job['relevant_jobs'] * _weights['relevant_jobs'] +
             job['relevant_degrees'] * _weights['relevant_degrees'] +
             job['location_match'] * _weights['location_match'])
    return score


def sort_jobs_by_score(jobs):
    for result in jobs:
        result['final_score'] = _calculate_job_score(result)

    # Sort job results by final score in descending order
    job_results = sorted(jobs, key=lambda x: x['final_score'], reverse=True)
    return job_results