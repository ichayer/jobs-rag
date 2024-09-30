# Define applicant's profile as JSON to test
MOCK_CV_EXTRACTED_DATA =  {
    "name": "GUSTAVO GARRIDO",
    "email" : "ingindustrial.gustavo@gmail.com",
    "address": "Wenceslao de tata 4672, caseros, Provincia de buenos aires",
    "city": "",
    "professional_experience_in_years": "",
    "highest_education": "INGENIERO INDUSTRIAL, PROMOCIÓN DE 2012 (Universidad del Magdalena - Colombia) and ESPECIALISTA EN INGENIERÍA CALIDAD, PROMOCIÓN DE 2018 (UNIVERSIDAD TECNOLÓGICA NACIONAL - ARGENTINA)",
    "skills": ["Node JS", "Angular", "Git", "SLQ en motores PL/SQL, MySQL, SQL SERVER", "Typescript", "Javascript", "HTML", "CSS", "Procesos ETL", "QLiksense"],
    "applied_for_profile": "",
    "education": [
        {
            "institute_name": "UNIVERSIDAD DEL MAGDALENA - COLOMBIA",
            "degree": "INGENIERO INDUSTRIAL, PROMOCIÓN DE 2012"
        },
        {
            "institute_name": "UNIVERSIDAD TECNOLÓGICA NACIONAL - ARGENTINA",
            "degree": "ESPECIALISTA EN INGENIERÍA CALIDAD, PROMOCIÓN DE 2018"
        }
    ],
    "professional_experience": [
        {
            "organisation_name": "HOSPITAL SAN JUAN DE DIOS",
            "duration": "JUNIO 2016 - PRESENTE",
            "profile": "COORDINADOR DE PROCESOS Y LIDER DE IMPLEMENTACIÓN. Análisis y relevamiento técnico para poder llevar a cabo la transformación digital de procesos."
        },
        {
            "organisation_name": "GESITIÓN 360",
            "duration": "",
            "profile": "FULL STACK DEVELOPER, 2019"
        }
    ]
}

# Define a prompt template to compare the applicant's profile to the job description
MATCH_APPLICATION_WITH_JOB_PROMPT = """
You are given the following applicant profile and job description. 
For each job description, I want you to:

1. Count how many skills from the applicant's profile match the skills required in the job description.
2. Count how many required skills are missing in the applicant's profile.
3. Check the applicant's previous jobs and count how many are relevant for the applied position.
4. Check the applicant's education and count how many degrees match the job's education requirements.
5. Rate the location match from 0 to 5, where 0 means the location is too far away, and 4 means the applicant is in the same zone or can easily commute. 5 means the job is fully remote. If the job is hybrid or fully on site use the scale 1 to 4
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

Just answer with a list of JSON files of the provided format and nothing else. No more extra text nor explanation, the output is aimed to be parsed as a valid json

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