
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from src.prompt import promptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langsmith import traceable
import os


## Document Loader
@traceable(name="load_documents", metadata={'Document_Loader': 'PyPDFLoader'})
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
@traceable(name="load_minimal_docs")
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
@traceable(name="text_split", metadata={'Text_splitter': 'RecursiveCharacterTextSplitter'})
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
@traceable(name="download_embedding_model", metadata={'model': 'sentence-transformers/all-MiniLM-L6-v2', 'dimensions': 384})
def download_embedding_model():
    """Download the embedding model from HuggingFace having 384-dimensional embedding space.
    """

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    return embedding_model

## Function that loads docs-> splits text -> download embedding model
@traceable(name='setupPipeline', metadata={'description' : 'makes the data ready to be loaded in vectorStore'})
def setupPipeline():
    """
    Function to perform indexing tasks:
        1. Load Document
        2. Text Splitting
        3. Download Embedding model

    Returns:
        text_chunks: splitted text chunks
        embedings:  Embedding model
    """
    extracted_data = load_documents(data='data/')
    filter_data = load_minimal_docs(extracted_data)
    text_chunks = text_split(filter_data)

    embeddings = download_embedding_model()

    return text_chunks, embeddings

@traceable(name='create_vectorStore')
def create_vectorStore(index_name):
    """
    Create the vectorStore in Pinecone

    Args:
        index_name (str): Index name of the vector store
        pinecone_api_key (str): Pinecone API KEY

    Returns:
        vectorStore: vectorStore with all documents
    """
    ## Creating a VectorStore
    pc = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))

    # index_name = "medical-chatbot"
    # existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    # if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1'),
    )

    ## Fetch the text_chunks & embedding model
    text_chunks, embeddings = setupPipeline()

    ## Load the records in pinecone

    vectorStore = PineconeVectorStore.from_documents(
        documents = text_chunks,
        embedding = embeddings,
        index_name = index_name
    )

    return vectorStore

@traceable(name='load_VectorStore')
def load_vectorStore(index_name, pinecone_api_key):
    """Load the vectorStore based on index_name

    Args:
        index_name (str): index_name with which the vectorStore is present in Pinecone
        pinecone_api_key (str): Pinecone API KEY

    Returns:
        vectorStore: VectorStore with all documents 
    """
    pc = Pinecone(api_key = pinecone_api_key)

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print("Index does not exist. Creating...")
        return create_vectorStore(index_name)
    
    print("Index exists. Loading only.")
    embeddings = download_embedding_model()
    return PineconeVectorStore.from_existing_index(index_name, embedding=embeddings)


## Retriever
@traceable(name='load_retriever', metadata={'Technique': 'Maximal Marginal Relevance (MMR)'})
def load_retriever(vectorStore):
    """
    Create the MMR retriever to query the vectorStore

    """
    retriever = vectorStore.as_retriever(search_type='mmr', lambda_mult=0.5, k=3)

    return retriever

## Format the retrieved docs
@traceable(name='format_docs', metadata={'description': 'Format the retrieved documents to be passed as context'})
def format_docs(docs):
    """Format the retrieved documents from the vectorStore which can be passed as an input to the LLM

    Args:
        docs (List(str)): List of relevant documents

    Returns:
        docs(str): Relevant documents in specific format
    """
    return "\n".join([d.page_content for d in docs])

## LLM 
@traceable(name='load_LLM', metadata={'model' : 'HuggingFaceH4/zephyr-7b-beta'})
def load_LLM():
    """
    Load the LLM model from huggingface

    """
    model = HuggingFaceEndpoint(
        repo_id = "HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        temperature = 0.2,
        max_new_tokens=256,
        top_p = 0.9,
    )
    llm = ChatHuggingFace(llm=model)
    
    return llm

## Chat Pipeline
@traceable(name='chat_pipeline')
def chat_pipeline():
    """
    Function to create the entire pipeline 
    
    """
    vectorStore = load_vectorStore('medical-chatbot', os.getenv("PINECONE_API_KEY"))
    retriever = load_retriever(vectorStore)
    prompt_template = promptTemplate()
    ## Output Parser
    parser = StrOutputParser()

    llm = load_LLM()

    retriever_chain = RunnableParallel({
            'input' : RunnablePassthrough(),
            'context' : retriever | RunnableLambda(format_docs)
        })

    # prompt template as Runnable
    prompt_runnable = RunnableLambda(
        lambda inputs: prompt_template.format_messages(
            input=inputs['input'], 
            context=inputs['context']
        )
    )

    main_chain = retriever_chain | prompt_runnable | llm | parser

    return main_chain
