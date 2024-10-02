import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

class VectorDatabase:
    def __init__(self, index_name, pinecone_api_key):
        self.index_name = index_name
        self.pinecone_api_key = pinecone_api_key
        self.vectorstore = self.__get_vector_store()
        self.embedding_size = 384

    def __get_vector_store(self):
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': os.environ["COMPUTE_DEVICE"]}
        )
        pc = Pinecone(api_key=self.pinecone_api_key)

        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=self.embedding_size,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

        return PineconeVectorStore(index_name=self.index_name, embedding=embeddings)

    def search(self, query, k=5):
        return self.vectorstore.search(
            query=query,
            search_type="similarity_score_threshold",
            k=k
        )

    def add_documents(self, documents):
        self.vectorstore.add_documents(documents)
