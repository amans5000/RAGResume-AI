import os
import chromadb


# ==========================================
# CHROMADB CLIENT
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# ==========================================
# COLLECTION
# ==========================================

collection = client.get_or_create_collection(
    name="resume_collection",
    metadata={
        "hnsw:space": "cosine"
    }
)


# ==========================================
# STORE DOCUMENTS
# ==========================================

def store_documents(
    chunks,
    embeddings,
    source_file
):
    """
    Store document chunks, embeddings
    and metadata in ChromaDB.
    """

    ids = []

    documents = []

    metadatas = []

    for i, chunk in enumerate(chunks):

        # Create unique ID
        chunk_id = (
            f"{source_file}_chunk_{i}"
        )

        ids.append(chunk_id)

        documents.append(
            chunk["text"]
        )

        metadatas.append(
            {
                "source": source_file,
                "chunk": i,
                "section": chunk["section"]
            }
        )

    # Upsert prevents duplicate-ID errors
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )


# ==========================================
# GET COLLECTION
# ==========================================

def get_collection():

    return collection


# ==========================================
# COLLECTION COUNT
# ==========================================

def get_collection_count():

    return collection.count()


# ==========================================
# CLEAR COLLECTION
# ==========================================

def clear_collection():
    """
    Deletes and recreates the collection to wipe existing indexed data.
    """
    global collection
    client.delete_collection("resume_collection")
    collection = client.get_or_create_collection(
        name="resume_collection",
        metadata={
            "hnsw:space": "cosine"
        }
    )
    return collection