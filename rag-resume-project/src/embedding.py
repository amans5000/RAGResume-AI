from sentence_transformers import SentenceTransformer


# Embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def create_embeddings(chunks):
    """
    Create embeddings for document chunks.

    The section name is included in the
    embedding text to provide additional context.
    """

    texts = []

    for chunk in chunks:

        text_for_embedding = (
            f"Section: {chunk['section']}\n"
            f"{chunk['text']}"
        )

        texts.append(
            text_for_embedding
        )

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings


def create_query_embedding(query):
    """
    Create embedding for the user query.
    """

    embedding = model.encode(
        [query]
    )

    return embedding[0]