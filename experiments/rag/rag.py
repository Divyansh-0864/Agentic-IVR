#!/usr/bin/env python3
"""
Tiny LangChain RAG prototype for IVR using Groq LLM and (preferably) local Hugging Face embeddings.

- Creates dummy IVR docs
- Builds embeddings (Hugging Face by default, OpenAI as fallback)
- Indexes docs in FAISS
- Interactive REPL: user query -> retrieve -> generate answer using Groq LLM

Set USE_LOCAL_EMB=0 to use OpenAI embeddings instead of Hugging Face.
Ensure GROQ_API_KEY is set to use the Groq LLM.
"""

from typing import List, Optional
import os
from uuid import uuid4
from dotenv import load_dotenv

# langchain imports
import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()
# -------------------------
# Dummy IVR documents
# -------------------------
document_1 = Document(
    page_content=(
        "To check your account balance, say 'check my balance' or press 1. "
        "You can hear recent transactions for the last 30 days by asking 'recent transactions'. "
        "For security, we will ask you to confirm your last 4 digits of your account number."
    ),
    metadata={"source": "acct", "title": "Account balance and recent transactions"},
)

document_2 = Document(
    page_content=(
        "To transfer money between accounts, say 'transfer' followed by amount and destination. "
        "For example, 'transfer five hundred to savings'. Transfers within the same bank are instant; "
        "external transfers may take up to 2 business days."
    ),
    metadata={"source": "transfer", "title": "Money transfer between accounts"},
)

document_3 = Document(
    page_content=(
        "If you forgot your PIN, say 'reset PIN'. We will send a one-time verification code to your "
        "registered mobile. After verifying, you can set a new MPIN. Do not share verification codes "
        "with anyone."
    ),
    metadata={"source": "pin", "title": "Reset PIN and authentication"},
)

document_4 = Document(
    page_content=(
        "If your card is lost or stolen, say 'block my card' immediately. We will block the card and "
        "issue a replacement. Replacement cards usually arrive in 5–7 business days."
    ),
    metadata={"source": "card", "title": "Card lost or stolen"},
)

document_5 = Document(
    page_content=(
        "Our branch hours are Monday through Friday, 9 AM to 5 PM, and Saturday 9 AM to 1 PM. "
        "For support, say 'connect to agent' to request a callback or visit our website for branch locations."
    ),
    metadata={"source": "branch", "title": "Branch hours and contact"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
]

uuids = [str(uuid4()) for _ in range(len(documents))]


# -------------------------
# Embeddings + FAISS index creation
# -------------------------
def build_vector_store(docs: List[Document], persist_dir: Optional[str] = None, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Split docs into chunks, create HF embeddings, build FAISS vector store,
    add docs with generated UUIDs, and return the vectorstore object.
    """
    # 1) Split docs into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs: List[Document] = []
    for doc in docs:
        parts = splitter.split_text(doc.page_content)
        for i, part in enumerate(parts):
            md = dict(doc.metadata)
            md["chunk"] = i
            split_docs.append(Document(page_content=part, metadata=md))

    print(f"[info] Created {len(split_docs)} chunks from {len(docs)} documents.")

    # 2) Embeddings (HuggingFace local)
    hf_model = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[info] Creating HuggingFaceEmbeddings({hf_model}) ...")
    embeddings = HuggingFaceEmbeddings(model_name=hf_model)

    # 3) Build FAISS vectorstore using LangChain helper
    print("[info] Building FAISS index from documents (this will compute embeddings)...")
    vector_store = FAISS.from_documents(split_docs, embeddings)

    # 4) Optionally persist
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        vector_store.save_local(persist_dir)
        print(f"[info] Saved FAISS index to {persist_dir}")

    # 5) If you need to add documents with explicit ids (demonstration),
    #    create uuids and call add_documents. Here we already used from_documents which handled adding.
    #    But to follow your requested pattern, show how to add extra docs:
    # uuids = [str(uuid4()) for _ in range(len(split_docs))]
    # vector_store.add_documents(documents=split_docs, ids=uuids)

    return vector_store



def retrieve_context(query: str, vector_store: FAISS) -> str:
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# -------------------------
# CLI / demo
# -------------------------
def main(persist_dir: Optional[str] = None):
    # Build vector store
    vs = build_vector_store(documents, persist_dir=persist_dir)
    llm = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant", max_tokens=500)

    print("\nRAG ready. Ask IVR questions (type 'exit').")
    while True:
        q = input("\nQuery> ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        try:
            ctx, ans = retrieve_context(q, vs)
        except Exception as e:
            print(f"[error] {e}")
            continue

        print("\n=== Answer ===\n")
        message=[
            ('system', "You are an IVR assistant helping users with their banking queries."),
            ('human', f"Context:\n{ctx}\n\nQuestion: {q}\nProvide a concise answer based on the context.")
        ]
        ans = llm.invoke(message)
        print(ans.content)

if __name__ == "__main__":
    os.getenv("GROQ_API_KEY") 
    main()

