import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

def get_vector_store(index_name):
  """ Creates vector store from Pinecone for storing and managing embeddings.

    :param str index_name: The name of the index to create or retrieve from Pinecone
    :return: PineconeVectorStore: An object representing the vector store in Pinecone for managing embeddings.

    :raise: ValueError: If the index creation fails due to invalid parameters or connection issues.
  """

  embeddings = HuggingFaceEmbeddings(  # embedding=OpenAIEmbeddings() rate limit
      model_name='sentence-transformers/all-MiniLM-L6-v2',
      model_kwargs={'device': 'cpu'}
  )

  embedding_size = 384

  pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])  # Pinecone is initialized using an API key stored in the environment variable

  if index_name not in pc.list_indexes().names():        # Check whether an index with the given index_name already exists
      pc.create_index(
          name=index_name,          # Name of the index
          dimension=embedding_size, # Size of the vectors (embeddings)
          metric="cosine",          # Distance metric used to compare vectors
          spec=ServerlessSpec(      # Determines the infrastructure used
              cloud='aws',          # Specifies that the Pinecone index is hosted on AWS
              region='us-east-1'    # Specifies the region of the cloud provider
          )
      )

  vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings) # initializes a PineconeVectorStore object using the index_name and the provided embeddings model or function

  return vectorstore