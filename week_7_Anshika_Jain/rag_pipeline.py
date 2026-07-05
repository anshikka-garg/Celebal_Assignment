import os

from dotenv import load_dotenv
import cohere

# pinecone → vector database ke liye
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset

load_dotenv()

COHERE_API_KEY    = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY")
INDEX_NAME        = os.getenv("PINECONE_INDEX_NAME", "rag-index")

co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

def load_data():
    print("Loading dataset from HuggingFace...")
    
    dataset = load_dataset(
        "rajpurkar/squad",
        split="train[:500]"
    )

    documents = []
    seen = set()
    
    for row in dataset:
        context = row.get('context', '')
        if context and context not in seen:
            documents.append(context)
            seen.add(context)

    print(f"Loaded {len(documents)} documents!")
    return documents
def chunk_text(documents, chunk_size=500, overlap=50):
    print(" Chunking documents...")

    chunks = []

    for doc in documents:
        doc_len = len(doc)
        start = 0
        while start < doc_len:
            end = start + chunk_size
            chunk = doc[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap

    print(f" Total Chunks Created: {len(chunks)}")
    return chunks

def create_embeddings(chunks):
    print("Creating embeddings...")
    
    response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    
    embeddings = response.embeddings
    print(f" Embeddings Created: {len(embeddings)}")
    return embeddings


def setup_pinecone(embeddings, chunks):
    print(" Setting up Pinecone...")

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f" Index '{INDEX_NAME}' created!")
    else:
        print(f" Index '{INDEX_NAME}' already exists!")

    index = pc.Index(INDEX_NAME)

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks     = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        vectors = [
            {
                "id":     f"chunk_{i+j}",
                "values": batch_embeddings[j],
                "metadata": {"text": batch_chunks[j]}
            }
            for j in range(len(batch_chunks))
        ]

        index.upsert(vectors=vectors)
        print(f" Uploaded batch {i//batch_size + 1}")

    print(" All chunks stored in Pinecone!")
    return index

def retrieve_context(query, index, top_k=5):
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

    context_chunks = [
        match["metadata"]["text"]
        for match in results["matches"]
    ]

    return context_chunks


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


def answer_question(query, index):
    context_chunks = retrieve_context(query, index)
    answer = generate_answer(query, context_chunks)
    return answer, context_chunks

def initialize_rag():
    print("Initializing RAG Pipeline...")
    
    documents = load_data()
    chunks = chunk_text(documents)
    embeddings = create_embeddings(chunks)
    index = setup_pinecone(embeddings, chunks)
    
    print("RAG Pipeline Ready!")
    return index

if __name__ == "__main__":
    index = initialize_rag()
    
    print("\n--- Testing RAG Pipeline ---")
    
    test_queries = [
        "What is architecture?",
        "What is a university?",
        "Who is a scientist?"
    ]
    
    for query in test_queries:
        answer, context = answer_question(query, index)
        print(f"\nQuestion: {query}")
        print(f"Answer: {answer}")
        print(f"Source Chunks: {len(context)}")
        print("-" * 50)