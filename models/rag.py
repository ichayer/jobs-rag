from pdfminer.high_level import extract_text
from indexing_pipeline.llm_handler import LLMHandler
from indexing_pipeline.vector_db import VectorDatabase
from utils.utils import load_env_vars


class RAG:
    """
    A singleton to handle the retrieval-augmented generation (RAG) process for
    matching job descriptions with applicant profiles.

    This class extracts text from an applicant's PDF resume, processes the text
    using a large language model (LLM) to generate a query, and retrieves relevant
    job descriptions from a vector database.

    Attributes:
        input_path (str): The file path to the applicant's resume (PDF).
        env_vars (dict): Environment variables for accessing external resources.
        llm_handler (LLMHandler): Handler for interacting with the LLM to process text.
        vector_db (VectorDatabase): Interface for interacting with Pinecone vector database.
    """

    _instance = None

    _weights = {
        "matching_skills": 0.4,
        "missing_skills": -0.1,  # Penalize missing skills
        "relevant_jobs": 0.25,
        "relevant_degrees": 0.1,
        "location_match": 0.15
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAG, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.env_vars = load_env_vars(["INDEX_NAME", "PINECONE_API_KEY"])
        self.llm_handler = LLMHandler()
        self.vector_db = VectorDatabase(index_name=self.env_vars["INDEX_NAME"], pinecone_api_key=self.env_vars["PINECONE_API_KEY"])

    def __process_applicant_profile(self, pdf_path):
        """
        Processes the text from the applicant profile to extract relevant data using the LLMHandler.
        """
        print("----------------------------------------------------------------------------------------------------")
        print("Processing applicant profile...")
        print("Extracting text from PDF file... ", end='', flush=True)
        text = extract_text(pdf_file=pdf_path)
        print("✅")
        print("Interpreting profile text with LLM... ", end='', flush=True)
        applicant_profile, raw_output = self.llm_handler.extract_data(text=text)
        print("✅")
        print(f"Raw model output:\n========START MODEL OUTPUT========\n{raw_output}\n=========END MODEL OUTPUT=========")
        return applicant_profile

    def __prepare_query(self, applicant_profile):
        """
        Prepares a query based on the applicant's profile data.
        """
        return self.llm_handler.prepare_query(applicant_profile=applicant_profile)

    def __search_jobs(self, query, k):
        """
        Search for job descriptions based on the applicant's query using the vector database.
        """
        return self.vector_db.search(query=query, k=k)

    def __calculate_job_score(self, job):
        """
        Calculate the final score for a given job based on predefined weights.
        """
        score = (job['matching_skills'] * self._weights['matching_skills'] +
                 job['missing_skills'] * self._weights['missing_skills'] +
                 job['relevant_jobs'] * self._weights['relevant_jobs'] +
                 job['relevant_degrees'] * self._weights['relevant_degrees'] +
                 job['location_match'] * self._weights['location_match'])
        return score

    def __sort_jobs_by_score(self, jobs):
        """
        Sort the list of jobs by their calculated score in descending order.
        :param jobs: A list of job descriptions with associated scores.
        """
        for job in jobs:
            job['final_score'] = self.__calculate_job_score(job=job)
        return sorted(jobs, key=lambda x: x['final_score'], reverse=True)

    def run(self, pdf_path, k=5):
        """
        Executes the entire RAG process: extracts the applicant's profile, generates
        a query, and retrieves job descriptions.
        :param pdf_path: The file path to the applicant's resume (PDF).
        :param k: The number of job descriptions to retrieve.
        """
        applicant_profile = self.__process_applicant_profile(pdf_path=pdf_path)
        applicant_query = self.__prepare_query(applicant_profile=applicant_profile)
        search_results = self.__search_jobs(query=applicant_query, k=k)
        return applicant_profile, applicant_query, search_results

    def run_with_scores(self, pdf_path, k=5):
        """
        Runs the RAG process and scores the retrieved job descriptions based on the
        applicant's profile. Returns the best match and the sorted job list.
        :param pdf_path: The file path to the applicant's resume (PDF).
        :param k: The number of job descriptions to retrieve.
        """
        applicant_profile, applicant_query, search_results = self.run(k=k, pdf_path=pdf_path)
        comparison_result, raw_output = self.llm_handler.compare_applicant_with_jobs(applicant_profile=applicant_profile, job_descriptions_text=search_results)
        print("----------------------------------------------------------------------------------------------------")
        print(f"Raw model output:\n========START MODEL OUTPUT========\n{raw_output}\n=========END MODEL OUTPUT=========")
        sorted_jobs = self.__sort_jobs_by_score(jobs=comparison_result)
        best_match = sorted_jobs[0]
        return best_match, sorted_jobs
