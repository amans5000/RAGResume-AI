import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

# Allow Python to import files from src/
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT RAG COMPONENTS
# ============================================================

from document_loader import load_document
from chunker import chunk_text
from embedding import create_embeddings

from vector_store import (
    store_documents,
    clear_collection,
    get_collection_count
)

from retriever import retrieve_context

from llm import generate_answer


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Resume RAG Assistant",
    page_icon="",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "processed" not in st.session_state:
    st.session_state.processed = False

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.title("Resume RAG Assistant")

st.markdown(
    """
Upload your resume and ask questions about it.

The system will:

**Upload → Extract → Chunk → Embed → ChromaDB → Retrieve → Groq**
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Resume")

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx"],
        help="Supported formats: PDF and DOCX"
    )

    st.divider()

    if uploaded_file is not None:

        st.write("### Selected file")

        st.write(
            f"`{uploaded_file.name}`"
        )

        process_button = st.button(
            "Process Resume",
            use_container_width=True
        )

    else:

        process_button = False

    st.divider()

    st.write("### Database")

    current_count = get_collection_count()

    st.write(
        f"Stored chunks: **{current_count}**"
    )


# ============================================================
# PROCESS UPLOADED DOCUMENT
# ============================================================

if process_button:

    with st.status(
        "Processing resume...",
        expanded=True
    ) as status:

        try:

            # ------------------------------------------------
            # STEP 1: SAVE UPLOADED FILE
            # ------------------------------------------------

            st.write("Saving uploaded file...")

            data_dir = ROOT_DIR / "data"

            data_dir.mkdir(
                exist_ok=True
            )

            file_path = (
                data_dir / uploaded_file.name
            )

            with open(
                file_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )


            # ------------------------------------------------
            # STEP 2: LOAD DOCUMENT
            # ------------------------------------------------

            st.write("Extracting text from document...")

            raw_text = load_document(
                str(file_path)
            )

            if not raw_text.strip():

                raise ValueError(
                    "No text could be extracted "
                    "from the uploaded document."
                )

            st.write(
                f"Extracted "
                f"{len(raw_text)} characters."
            )


            # ------------------------------------------------
            # STEP 3: CHUNK DOCUMENT
            # ------------------------------------------------

            st.write(
                "Creating section-aware chunks..."
            )

            chunks = chunk_text(
                raw_text
            )

            if not chunks:

                raise ValueError(
                    "No chunks were created."
                )

            st.write(
                f"Created {len(chunks)} chunks."
            )


            # ------------------------------------------------
            # STEP 4: CREATE EMBEDDINGS
            # ------------------------------------------------

            st.write(
                "Creating embeddings..."
            )

            embeddings = create_embeddings(
                chunks
            )

            st.write(
                "Embeddings created."
            )


            # ------------------------------------------------
            # STEP 5: CLEAR OLD DOCUMENT
            # ------------------------------------------------

            st.write(
                "Clearing previous resume data..."
            )

            clear_collection()


            # ------------------------------------------------
            # STEP 6: STORE IN CHROMADB
            # ------------------------------------------------

            st.write(
                "Storing chunks in ChromaDB..."
            )

            store_documents(
                chunks,
                embeddings,
                uploaded_file.name
            )


            # ------------------------------------------------
            # UPDATE SESSION STATE
            # ------------------------------------------------

            st.session_state.processed = True

            st.session_state.file_name = (
                uploaded_file.name
            )

            st.session_state.chunk_count = (
                len(chunks)
            )

            # Clear previous conversation
            st.session_state.messages = []


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            status.update(
                label="Resume processed successfully!",
                state="complete",
                expanded=False
            )


        except Exception as e:

            status.update(
                label="Error processing resume",
                state="error",
                expanded=True
            )

            st.error(
                f"Error: {e}"
            )


# ============================================================
# CURRENT DOCUMENT STATUS
# ============================================================

if st.session_state.processed:

    st.success(
        f"Ready: "
        f"{st.session_state.file_name} "
        f"• "
        f"{st.session_state.chunk_count} chunks"
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.processed:

    user_query = st.chat_input(
        "Ask something about your resume..."
    )


    if user_query:

        # ----------------------------------------------------
        # DISPLAY USER MESSAGE
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.markdown(
                user_query
            )


        # Save user message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query
            }
        )


        # ----------------------------------------------------
        # RETRIEVAL + GENERATION
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching resume..."
                ):

                    # ----------------------------------------
                    # RETRIEVE CONTEXT
                    # ----------------------------------------

                    results = retrieve_context(
                        user_query,
                        top_k=4
                    )


                    documents = results.get(
                        "documents",
                        [[]]
                    )[0]

                    metadatas = results.get(
                        "metadatas",
                        [[]]
                    )[0]

                    distances = results.get(
                        "distances",
                        [[]]
                    )[0]


                    if not documents:

                        answer = (
                            "I couldn't find relevant "
                            "information in the resume."
                        )

                        st.markdown(
                            answer
                        )

                    else:

                        # ------------------------------------
                        # BUILD CONTEXT
                        # ------------------------------------

                        context_blocks = []


                        for i, document in enumerate(
                            documents
                        ):

                            metadata = (
                                metadatas[i]
                                if i < len(metadatas)
                                else {}
                            )

                            section = metadata.get(
                                "section",
                                "GENERAL"
                            )


                            context_blocks.append(
                                f"[Section: {section}]\n"
                                f"{document}"
                            )


                        combined_context = (
                            "\n\n---\n\n".join(
                                context_blocks
                            )
                        )


                        # ------------------------------------
                        # GENERATE ANSWER
                        # ------------------------------------

                        with st.spinner(
                            "Generating answer..."
                        ):

                            answer = generate_answer(
                                user_query,
                                combined_context
                            )


                        # ------------------------------------
                        # DISPLAY ANSWER
                        # ------------------------------------

                        st.markdown(
                            answer
                        )


                        # ------------------------------------
                        # SHOW RETRIEVED CONTEXT
                        # ------------------------------------

                        with st.expander(
                            "View retrieved context"
                        ):

                            for i, document in enumerate(
                                documents
                            ):

                                metadata = (
                                    metadatas[i]
                                    if i < len(metadatas)
                                    else {}
                                )

                                section = metadata.get(
                                    "section",
                                    "GENERAL"
                                )

                                distance = (
                                    distances[i]
                                    if i < len(distances)
                                    else None
                                )


                                st.markdown(
                                    f"### Context {i + 1}"
                                )

                                st.write(
                                    f"**Section:** "
                                    f"{section}"
                                )

                                if distance is not None:

                                    st.write(
                                        f"**Distance:** "
                                        f"{distance:.4f}"
                                    )

                                st.write(
                                    document
                                )

                                st.divider()


                # --------------------------------------------
                # SAVE ASSISTANT RESPONSE
                # --------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            except Exception as e:

                error_message = (
                    f"Error while generating answer: {e}"
                )

                st.error(
                    error_message
                )


# ============================================================
# NO DOCUMENT UPLOADED
# ============================================================

else:

    st.info(
        "Upload a PDF/DOCX resume from the sidebar "
        "and click **Process Resume** to begin."
    )