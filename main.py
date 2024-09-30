import json
import os
import re

from scores import sort_jobs_by_score
from langchain.chains.llm import LLMChain
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate

from pinecone_helper import get_vector_store
from pdfminer.high_level import extract_text
from json_helper import InputData as input
from dotenv import load_dotenv, find_dotenv

from string_vars import MOCK_CV_EXTRACTED_DATA, MATCH_APPLICATION_WITH_JOB_PROMPT

_ = load_dotenv(find_dotenv())

text = extract_text("cvs/CV-DESARROLLADOR.pdf")

llm = input.llm()


#applicant_profile = llm.invoke(input.input_data(text))

# this line is used to test with mocked data so that not to wait for the llm response
applicant_profile = MOCK_CV_EXTRACTED_DATA

# Convert the applicant's profile into a single text input
applicant_query = f"""
    Professional Experience: {", ".join([f"{exp['profile']}" for exp in applicant_profile['professional_experience']])} 
    Skills: {", ".join(applicant_profile['skills'])}
    Education: {", ".join([f"{edu['degree']}" for edu in applicant_profile['education']])}
"""

# Get the database
vectorstore = get_vector_store(os.environ['INDEX_NAME'])

# Search for the most relevant job descriptions in Pinecone
search_results = vectorstore.search(
    query=applicant_query,
    search_type="similarity_score_threshold",
    k=5  # Retrieve top 5 matching job descriptions
)


job_descriptions_text = "\n\n".join([f"Source: {result.metadata['source']}\nText: {result.page_content}" for result in search_results])

# Combine the applicant's profile and job descriptions into the prompt
prompt = PromptTemplate(input_variables=["address", "city", "professional_experience", "education", "skills", "job_description"], template=MATCH_APPLICATION_WITH_JOB_PROMPT)

llm = Ollama(model="llama3")

# Create the LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Prepare inputs for the prompt
inputs = {
    "address": applicant_profile['address'],
    "city": applicant_profile['city'],
    "professional_experience": ", ".join([f"{exp['profile']} at {exp['organisation_name']}" for exp in applicant_profile['professional_experience']]),
    "skills": ", ".join(applicant_profile['skills']),
    "education": ", ".join([f"{edu['degree']} from {edu['institute_name']}" for edu in applicant_profile['education']]),
    "job_description": job_descriptions_text
}

# Run the chain and get the structured response
comparison_result = chain.run(inputs)

# From a json string to a list of iterable maps with the data from the llm
comparison_result = json.loads(comparison_result)

# Calculate a score from the given response and order the jobs
sorted_jobs = sort_jobs_by_score(comparison_result)
best_match = sorted_jobs[0]

# Display the recommendation
print(f"Recommended job source: {best_match['source']} with a final score of {best_match['final_score']:.2f}")

# Display all scores
for result in sorted_jobs:
    print(f"Source: {result['source']}, Final Score: {result['final_score']:.2f}")