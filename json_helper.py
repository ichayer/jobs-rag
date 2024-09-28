from langchain_community.llms import Ollama

json_content = """{{
    "name": "",
    "email" : "",
    "address": "",
    "city": "",
    "professional_experience_in_years": "",
    "highest_education": "",
    "skills": ["",""],
    "applied_for_profile": "",
    "education": [
        {{
            "institute_name": "",
            "degree": ""
        }},
        {{
            "institute_name": "",
            "degree": ""
        }}
    ],
    "professional_experience": [
        {{
            "organisation_name": "",
            "duration": "",
            "profile": ""
        }},
        {{
            "organisation_name": "",
            "duration": "",
            "profile": ""
        }}
    ]
}}"""


class InputData:
    def input_data(text):

        input = f"""Extract relevant information from the following resume text and fill the provided JSON template. Ensure all keys in the template are present in the output, even if the value is empty or unknown. If a specific piece of information is not found in the text, use 'Not provided' as the value.

        Resume text:
        {text}

        JSON template:
        {json_content}

        Instructions:
        1. Carefully analyse the resume text.
        2. Extract relevant information for each field in the JSON template.
        3. If a piece of information is not explicitly stated, make a reasonable inference based on the context.
        4. Ensure all keys from the template are present in the output JSON.
        5. Format the output as a valid JSON string.

        Output the filled JSON template only, without any additional text or explanations."""
        
        return input
    
    def llm():
        llm = Ollama(model="llama3")
        return llm
