# jobs-rag

## Setup

Create a virtual environment:

```bash
> python -m venv venv
```

Activate the virtual environment:

```bash
> source venv/bin/activate
```

Install the dependencies:

```bash
> pip install -r requirements.txt
```

Fill .env file with Pinecone API key. There is a `.env.example` file to guide you.

## Requirements
You will also need to have Ollama running on your local computer. You can get ollama from [here](https://ollama.com/). Be mindful the current code uses the llama3 model (8 billion parameters). The authors are not be responsible for any broken computers.

In Windows, after executing the downloaded `.exe` file, open a Command Prompt and run the following command

```bash
ollama run llama3
```

## Running the code
1. Execute `load_jobs.py` to import job data from the directory specified by the `JOBS_FOLDER` environment variable into Pinecone. This action will generate a Pinecone index named according to the `INDEX_NAME` environment variable.
2. Run `recommend_jobs.py` to get recommendations for a job.