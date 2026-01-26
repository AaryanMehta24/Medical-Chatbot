from src.utils import chat_pipeline
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Annotated
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY



# Main function
main_chain = chat_pipeline()



class ChatInput(BaseModel):
    question: Annotated[str, Field(..., description="I'm a medical assistant how can i help you?")]

class ChatResponse(BaseModel):
    answer: str

app = FastAPI()


@app.post('/chat')
def chat(input_data : ChatInput):

    inputQuestion = input_data.model_dump()
   
    results = main_chain.invoke(inputQuestion["question"])
    
    return ChatResponse(answer=results)

