import os

from langchain_core.documents import Document


class DocumentLoader:
    def __init__(self, jobs_folder_path):
        self.jobs_folder_path = jobs_folder_path

    def load_job_descriptions(self):
        job_descriptions = []
        print("Loading document files:")
        for filename in os.listdir(self.jobs_folder_path):
            if filename.endswith(".txt"):
                print(f" - {filename} ", end='', flush=True)
                with open(os.path.join(self.jobs_folder_path, filename), "r") as file:
                    content = file.read()
                    doc = Document(page_content=content, metadata={"source": filename})
                    job_descriptions.append(doc)
                print("✅")
        return job_descriptions
