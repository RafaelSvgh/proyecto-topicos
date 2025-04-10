from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import os

OPENAI_API_KEY = ""

def cargar_base_vectorial():
    with open("conocimiento.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Fragmentar por párrafos (\n\n)
    chunks = raw_text.split("\n\n")
    docs = [Document(page_content=chunk.strip()) for chunk in chunks if chunk.strip()]

    # Inicializar embeddings y ChromaDB
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./db")

    vectordb.persist()
    return vectordb
