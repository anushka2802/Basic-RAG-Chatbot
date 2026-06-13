# rag_chatbot.py — Production-ready RAG in <50 lines
# pip install langchain langchain-openai langchain-community faiss-cpu pypdf

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()   # load .env variables


# ── Config ────────────────────────────────────────────────
api_key = os.getenv("GROQ_API")

MODEL = "openai/gpt-oss-120b"
DOCS_PATH   = "ZariB2BCollection.pdf"   # swap with any PDF / txt
TOP_K       = 4                     # chunks retrieved per query
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 50

# ── 1. Load ───────────────────────────────────────────────
loader = PyPDFLoader(DOCS_PATH)
documents = loader.load()

# ── 2. Split ──────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_documents(documents)

# ── 3. Embed + Store ──────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")   # persist to disk

# ── 4. Retrieval Chain ────────────────────────────────────
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

prompt = PromptTemplate.from_template("""
Use the context below to answer the question.
If you don't know, say "I don't know" — don't make things up.

Context: {context}
Question: {question}
Answer:""")

llm = llm = ChatGroq(api_key=api_key, model=MODEL, temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        result = qa_chain.invoke({"query": query})
        print(f"\n💬 {result['result']}")
