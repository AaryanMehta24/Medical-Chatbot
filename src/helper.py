
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.docstore.document import Document

## Document Loader
def load_documents(data):
    """To load the documents from the external knowledge base

    Args:
        data (.pdf): External Knowledge base

    Returns:
        documents: External knowledge base is loaded in the form of list of documents
    """

    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

## Filter Documents
def load_minimal_docs(documents: List[Document]) -> List[Document]:
    """Function to filter the information that each list of document holds.

    Args:
        documents (List[Document]): List of documents

    Returns:
        List[Document]: List of documents contain info of only the [page_content, metadata-source]
    """

    minimal_docs = []
    for doc in documents:
        src = doc.metadata.get("source")
        minimal_doc = Document(page_content=doc.page_content, metadata={"source": src})
        minimal_docs.append(minimal_doc)

    return minimal_docs

## Text Splitter
def text_split(documents):
    """Function to break down text into smaller chunks using MMR technique

    Args:
        documents (List[Document]): List of documents

    Returns:
        docs: documents split into smaller chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    docs = text_splitter.split_documents(documents)

    return docs

## Embedding model to generate embeddings of the text chunks
def download_embedding_model():
    """Download the embedding model from HuggingFace having 384-dimensional embedding space.
    """

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    return embedding_model
 