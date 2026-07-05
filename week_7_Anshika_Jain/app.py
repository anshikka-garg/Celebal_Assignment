import streamlit as st
import os
from dotenv import load_dotenv
import cohere
from pinecone import Pinecone
from datasets import load_dataset

load_dotenv()

COHERE_API_KEY   = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "rag-index")

co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def retrieve_context(query, top_k=5):
    query_embedding = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return [match["metadata"]["text"] for match in results["matches"]]


def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    message = f"""Answer the question based only on the context provided below.

Context:
{context}

Question: {query}

Answer:"""

    response = co.chat(
        model="command-a-03-2025",
        message=message,
        temperature=0.3
    )

    return response.text.strip()


st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Question Answering System")
st.subheader("RAG Pipeline — Cohere + Pinecone")

st.markdown("""
This system answers questions using **Retrieval-Augmented Generation (RAG)**:
- Retrieves relevant document chunks from Pinecone
- Generates accurate answers using Cohere LLM
""")

st.divider()

query = st.text_input(
    "Ask a question:",
    placeholder="e.g. What is architecture?"
)

if st.button("Get Answer") and query:
    with st.spinner("Retrieving context and generating answer..."):
        context_chunks = retrieve_context(query)
        answer = generate_answer(query, context_chunks)

    st.subheader("Answer:")
    st.write(answer)

    with st.expander("View Retrieved Context Chunks"):
        for i, chunk in enumerate(context_chunks):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(chunk)
            st.divider()

st.sidebar.title("About")
st.sidebar.markdown("""
**RAG Document Q&A System**

**Stack:**
- Cohere Embeddings
- Pinecone Vector DB
- Streamlit UI

**Dataset:**
- SQuAD (Stanford Q&A)

**Internship:** Celebal Technologies
**Week:** 7
""")