import time
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
import os
from pydantic import BaseModel

# import google.generativeai as genai
from langchain.chat_models import init_chat_model

from fastapi.middleware.cors import CORSMiddleware




class AskRequest(BaseModel):
    query: str

rag_chain = None
app_rag = None

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"{doc.metadata['url']}\n{doc.page_content}" for doc in docs)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"] if serving from local file / Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
def startup_event():
    global rag_chain
    global app_rag

    if not os.environ.get("GOOGLE_API_KEY"):
        # os.environ["GOOGLE_API_KEY"] = "AIzaSyDmAcinTsUXaah7faYxcuvyr5v7DZ0R_f4"  AIzaSyD0L2zheH0V19aQKo5hqWvJsWH2Muxgxrs
        os.environ["GOOGLE_API_KEY"] = "AIzaSyD0L2zheH0V192Muxgxrs"

    # genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    
    client = QdrantClient(
    url="https://637023ed-a0a7-45c9-b4a4-22fb60fc4ae7.europe-west3-0.gcp.cloud.qdrant.io", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJyIn0.i2Am7zbIrlT8UROWlfOeQZyaxtSsb88tUpnDglDbtbs"
)
    embeddings = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-modernbert-base")

    vector_store = QdrantVectorStore(
        collection_name="reddit_vectors",
        embedding=embeddings,
        client=client
    )
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(
             kwargs={"k":"8"}
        ),
        llm=llm
    )

    prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", """|
                            you are ask askPESU, a helpful assistant developed by the PESU students that helps users find information about PES University.
                            You must answer in detail using knowledge only from the r/PESU subreddit, which will be provided to you as context.
                            Always answer accurately, comprehensively and in depth, based on the context, optionally formatted in markdown.
                            Remember to end your response by citing the sources of your information in the format "Sources:\n\n<link1>\n<link2>".
                            If the context does not contain the answer, respond with "I'm sorry, I don't have that information."
                            If asked anything unrelated to PES University, politely decline.
                    """),
                    ("human", "Question: {question}\nContext: {context}\nAnswer:"),
                ]
            )
    
    app_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", """|
                            you must answer "I am askPESU, a helpful assistant developed by the PESU students that helps users find information about PES University" when someone asks you who are you.
                            You must answer to the point using knowledge only from the r/PESU subreddit, which will be provided to you as context.
                            Always answer accurately, based on the context, and do not format in markdown.
                            If the context does not contain the answer, respond with "I'm sorry, I don't have that information."
                            If asked anything unrelated to PES University, politely decline.
                    """),
                    ("human", "Question: {question}\nContext: {context}\nAnswer:"),
                ]
            )
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    app_rag = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | app_prompt
        | llm
        | StrOutputParser()
    )
    print("Rag initialised")

@app.get("/")
def alive():
    return FileResponse("index.html")


@app.post("/ask")
async def generate_response(query: AskRequest):
        print(query.query)
        answer = rag_chain.invoke(query.query)
        print(answer)
        return {"answer": answer}

@app.post("/app")
async def generate_response(query: AskRequest):
        print(query.query)
        answer = app_rag.invoke(query.query)
        print(answer)
        return {"answer": answer}
         
