from dotenv import load_dotenv
import os
from src.helper import load_documents, load_minimal_docs, text_split, download_embedding_model
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY

extracted_data = load_documents(data='data/')
filter_data = load_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embeddings = download_embedding_model()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key = pinecone_api_key)

## Creating a VectorStore

index_name = "medical-chatbot"
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1'),
    )

index = pc.Index(index_name)

## Load the records in pinecone

vectorStore = PineconeVectorStore.from_documents(
    documents = text_chunks,
    embedding = embeddings,
    index_name = index_name
)

