## Prompt Template
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def promptTemplate():
    system_prompt = (
        "You are an Medical assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "Context:\n{context}"  
    )
    prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),    
])

    return prompt_template
