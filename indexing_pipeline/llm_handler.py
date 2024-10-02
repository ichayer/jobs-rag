import json
from langchain.chains.llm import LLMChain
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate

# NICE: Perhaps implement a Singleton pattern for the LLMHandler class
class LLMHandler:

    # Define a prompt template to compare the applicant's profile to the job description
    MATCH_APPLICATION_WITH_JOB_PROMPT = """
    You are given the following applicant profile and job description.
    For each job description, I want you to:

    1. Count how many skills from the applicant's profile match the skills required in the job description.
    2. Count how many required skills are missing in the applicant's profile.
    3. Check the applicant's previous jobs and count how many are relevant for the applied position.
    4. Check the applicant's education and count how many degrees match the job's education requirements.
    5. Rate the location match from 0 to 5, where 0 means the location is too far away, and 4 means the applicant is in the same zone or can easily commute. 5 means the job is fully remote. If the job is hybrid or fully on site use the scale 1 to 4.
    6. Retrieve de source of the job description so that it is identifiable.

    Return the result in JSON format with the following fields:
    - matching_skills
    - missing_skills
    - relevant_jobs
    - relevant_degrees
    - location_match (1-5 scale)
    - source

    Applicant Profile:
    Address: {address}
    City: {city}
    Professional Experience: {professional_experience}
    Skills: {skills}
    Education: {education}

    Job Description:
    {job_description}

    Just answer with a list of JSON files of the provided format and nothing else. No more extra text nor explanation, the output is aimed to be parsed as a valid JSON.

    Example of a valid answer:

    [{{
    "matching_skills": 7,
    "missing_skills": 2,
    "relevant_jobs": 1,
    "relevant_degrees": 0,
    "location_match": 4,
    "source": "software2.txt"
    }},
    {{
    "matching_skills": 6,
    "missing_skills": 3,
    "relevant_jobs": 0,
    "relevant_degrees": 0,
    "location_match": 5,
    "source": "software1.txt"
    }}]
    """

    def __init__(self):
        self.llm = Ollama(model="llama3")
        self.json_content = self._get_json_template()

    def _get_json_template(self):
        return """{{
            "name": "",
            "email": "",
            "address": "",
            "city": "",
            "professional_experience_in_years": "",
            "highest_education": "",
            "skills": ["",""],
            "applied_for_profile": "",
            "education": [
                {{"institute_name": "", "degree": ""}},
                {{"institute_name": "", "degree": ""}}
            ],
            "professional_experience": [
                {{"organisation_name": "", "duration": "", "profile": ""}},
                {{"organisation_name": "", "duration": "", "profile": ""}}
            ]
        }}"""

    def __cut_off_json_excess(self, text):
        start1 = text.find('{')
        start2 = text.find('[')
        end1 = text.rfind('}')
        end2 = text.rfind(']')

        if start1 != -1 and start2 != -1:
            start = min(start1, start2)
        else:
            start = max(start1, start2)

        if end1 != -1 and end2 != -1:
            end = max(end1, end2)
        else:
            end = max(end1, end2)

        if end != -1:
            text = text[:end+1]
        if start != -1:
            text = text[start:]

        return text

    # Sometimes the LLM model is not able to output correctly a JSON string
    def extract_data(self, text, max_retries=7):
        prompt = self.create_prompt(text)
        for attempt in range(max_retries):
            try:
                output = self.llm.invoke(prompt)
                output = self.__cut_off_json_excess(output)
                return json.loads(output)
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries}: Failed to decode JSON. LLM output JSON is malformed.")
        raise ValueError(f"Failed to decode JSON after {max_retries} attempts. LLM output JSON is malformed.")

    def prepare_query(self, applicant_profile):
        return f"""
            Professional Experience: {", ".join([exp['profile'] for exp in applicant_profile['professional_experience']])}
            Skills: {", ".join(applicant_profile['skills'])}
            Education: {", ".join([edu['degree'] for edu in applicant_profile['education']])}
        """

    def create_prompt(self, text):
        return f"""Extract relevant information from the following resume text and fill the provided JSON template. Ensure all keys in the template are present in the output, even if the value is empty or unknown. If a specific piece of information is not found in the text, use 'Not provided' as the value.

        Resume text:
        {text}

        JSON template:
        {self.json_content}

        Instructions:
        1. Carefully analyze the resume text.
        2. Extract relevant information for each field in the JSON template.
        3. If a piece of information is not explicitly stated, make a reasonable inference based on the context.
        4. Ensure all keys from the template are present in the output JSON.
        5. Format the output as a valid JSON string.

        Output the filled JSON template only, without any additional text or explanations.
        Be precise and sure to follow JSON syntax and structure correctly.
        """

    def compare_applicant_with_jobs(self, applicant_profile, job_descriptions_text):
        prompt = PromptTemplate(input_variables=["address", "city", "professional_experience", "education", "skills", "job_description"],
                                template=self.MATCH_APPLICATION_WITH_JOB_PROMPT)

        inputs = {
            "address": applicant_profile['address'],
            "city": applicant_profile['city'],
            "professional_experience": ", ".join([f"{exp['profile']} at {exp['organisation_name']}" for exp in applicant_profile['professional_experience']]),
            "skills": ", ".join(applicant_profile['skills']),
            "education": ", ".join([f"{edu['degree']} from {edu['institute_name']}" for edu in applicant_profile['education']]),
            "job_description": job_descriptions_text
        }

        chain = LLMChain(llm=self.llm, prompt=prompt)
        output = chain.run(inputs)
        output = self.__cut_off_json_excess(output)
        return output
