import argparse
import os
import sys
import warnings
from pathlib import Path

# Suppress harmless OpenSSL/urllib3 warnings on macOS
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")

# Ensure src directory is in sys.path for direct script execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from chunker import chunk_text
from document_loader import load_document
from embedding import create_embeddings
from llm import MODEL_NAME, generate_answer
from retriever import retrieve_context
from vector_store import (
    clear_collection,
    get_collection_count,
    store_documents,
)

DEFAULT_RESUME_PATH = os.path.join(PROJECT_ROOT, "data", "resume.pdf")


def index_document(file_path=None, clear_existing=False):
    """
    Load, chunk, embed, and store a resume document in ChromaDB.
    """
    if file_path is None:
        file_path = DEFAULT_RESUME_PATH

    resolved_path = os.path.abspath(file_path)

    if not os.path.exists(resolved_path):
        print(f"Error: Resume file not found at '{resolved_path}'")
        return False

    file_name = os.path.basename(resolved_path)
    print(f"\nLoading document: {file_name}")
    try:
        raw_text = load_document(resolved_path)
    except Exception as e:
        print(f"Failed to load document: {e}")
        return False

    print(f"Chunking text...")
    chunks = chunk_text(raw_text)
    if not chunks:
        print("No text chunks could be extracted from document.")
        return False

    sections = sorted(list(set(c["section"] for c in chunks)))
    print(f"Created {len(chunks)} chunks across sections: {', '.join(sections)}")

    if clear_existing:
        print("Clearing previous vector collection...")
        clear_collection()

    print("Generating embeddings (Sentence-Transformers all-MiniLM-L6-v2)...")
    embeddings = create_embeddings(chunks)

    print("Storing into persistent ChromaDB...")
    store_documents(chunks, embeddings, file_name)

    total_chunks = get_collection_count()
    print(f"Successfully indexed '{file_name}'! Total chunks in store: {total_chunks}\n")
    return True


def query_pipeline(query, top_k=4, verbose=False):
    """
    Retrieve matching context chunks and generate an answer using the LLM.
    """
    total_chunks = get_collection_count()
    if total_chunks == 0:
        print("Vector database is empty. Attempting auto-indexing...")
        if not index_document():
            return None, []

    results = retrieve_context(query, top_k=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return "No relevant information found in the indexed resume.", []

    context_blocks = []
    sources = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        sec = meta.get("section", "GENERAL")
        sources.append(sec)
        context_blocks.append(f"[Section: {sec}]\n{doc}")

    combined_context = "\n\n---\n\n".join(context_blocks)

    if verbose:
        print("\n--- Retrieved Context ---")
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            print(f"[{i+1}] Section: {meta.get('section')} (Cosine Dist: {dist:.4f})")
            print(f"    {doc[:120]}...\n")
        print("-------------------------\n")

    answer = generate_answer(query, combined_context)
    return answer, sources


def interactive_mode(top_k=4):
    """
    Run an interactive question-answering CLI session.
    """
    total = get_collection_count()
    if total == 0:
        print("No existing index found. Building index from default resume...")
        if not index_document():
            print("Cannot start interactive mode without an indexed document.")
            return

    print("=" * 60)
    print("RAG Resume Assistant")
    print("=" * 60)
    print(f"Chunks in ChromaDB : {get_collection_count()}")
    print(f"LLM Model          : {MODEL_NAME} (Groq)")
    print("Commands           : Type 'exit' or 'q' to quit")
    print("                   : Type 'reindex' to rebuild vector store")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nAsk a question about the resume: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if user_input.lower() in ["reindex", "index"]:
            index_document(clear_existing=True)
            continue

        print("Searching resume and generating answer...\n")
        try:
            answer, sources = query_pipeline(user_input, top_k=top_k)
            print("Answer:")
            print(answer)
            if sources:
                unique_sources = sorted(list(set(sources)))
                print(f"Sources: {', '.join(unique_sources)}")
        except Exception as e:
            print(f"Error generating answer: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Resume Assistant: Index and query resumes with semantic search."
    )
    parser.add_argument(
        "--index", "-i",
        action="store_true",
        help="Build or rebuild the vector index."
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to resume file (PDF or DOCX) to index."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Direct question to ask about the resume."
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=4,
        help="Number of relevant chunks to retrieve (default: 4)."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive Q&A session."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed retrieval logs and distances."
    )

    args = parser.parse_args()

    # Index command
    if args.index:
        success = index_document(file_path=args.file, clear_existing=True)
        sys.exit(0 if success else 1)

    # Single query command
    if args.query:
        answer, sources = query_pipeline(args.query, top_k=args.top_k, verbose=args.verbose)
        if answer is not None:
            print("Answer:")
            print(answer)
            if sources:
                unique_sources = sorted(list(set(sources)))
                print(f"Sources: {', '.join(unique_sources)}")
        return

    # Default or explicit interactive mode
    interactive_mode(top_k=args.top_k)


if __name__ == "__main__":
    main()