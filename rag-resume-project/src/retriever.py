from embedding import create_query_embedding
from vector_store import get_collection


def retrieve_context(
    query,
    top_k=5
):
    """
    Retrieve the most relevant chunks
    from ChromaDB.
    """

    collection = get_collection()

    # Create embedding for query
    query_embedding = create_query_embedding(
        query
    )

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "distances",
            "metadatas"
        ]
    )

    return results