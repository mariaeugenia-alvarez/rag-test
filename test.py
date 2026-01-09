"""
RAG Mínimo - Código esencial para cargar PDF y hacer consultas
pip install langchain langchain-anthropic langchain-huggingface langchain-community faiss-cpu pypdf python-dotenv sentence-transformers torch
"""

import os
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Cargar API key
load_dotenv()
# O configurar directamente: os.environ["OPENAI_API_KEY"] = "tu-api-key"

# 1. CARGAR PDF
loader = PyPDFLoader("pdf/ARRANQUE ALUMNOS IA4.pdf")
documents = loader.load()

# 2. DIVIDIR EN CHUNKS
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 3. CREAR EMBEDDINGS Y VECTORSTORE
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. CREAR RETRIEVER
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# 5. CREAR CHAIN RAG
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


prompt = ChatPromptTemplate.from_template(
    """
Responde la pregunta basándote en el siguiente contexto:

{context}

Pregunta: {question}

Respuesta:"""
)

llm = ChatAnthropic(model="claude-opus-4-1", temperature=0)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. HACER CONSULTAS
respuesta = rag_chain.invoke("¿De qué trata el documento?")
print(respuesta)
