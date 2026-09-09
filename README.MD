# ResumeRAG — AI-Powered Resume Question Answering System

A Retrieval-Augmented Generation (RAG) application that allows users to upload a resume in PDF or DOCX format and ask natural-language questions about its contents.

The system extracts text from the uploaded document, creates section-aware chunks, generates embeddings, stores them in ChromaDB, retrieves the most relevant information for a user's query, and uses Groq LLM to generate a context-grounded answer.

---

## Features

- Upload resumes in PDF and DOCX format
- Extract text from documents
- Section-aware document chunking
- Generate semantic embeddings using Sentence Transformers
- Store embeddings in ChromaDB
- Semantic similarity-based retrieval
- Generate answers using Groq LLM
- Interactive Streamlit web interface
- View retrieved context used for answer generation
- Display retrieved sections and similarity distances
- Replace the currently indexed resume with a new resume

---

## Architecture

The application follows a standard Retrieval-Augmented Generation pipeline.

```text
                         Streamlit Frontend
                                |
                                |
                         Upload Resume
                                |
                                v
                     +----------------------+
                     |  Document Loader     |
                     |                      |
                     | PDF / DOCX -> Text   |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | Section-Aware        |
                     | Chunking             |
                     |                      |
                     | Text -> Chunks       |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | Embedding Model      |
                     |                      |
                     | Chunks -> Vectors    |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     |      ChromaDB        |
                     |                      |
                     |   Vector Database    |
                     +----------+-----------+
                                |
                                |
                                | User Question
                                v
                     +----------------------+
                     | Query Embedding      |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | Semantic Retrieval   |
                     |                      |
                     |    Top-K Chunks      |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | Retrieved Context    |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     |      Groq LLM        |
                     |                      |
                     | Context + Question   |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     |    Final Answer      |
                     +----------------------+
```

---

## RAG Pipeline

The application is divided into two main phases:

1. Document ingestion
2. Question answering

### 1. Document Ingestion

When a resume is uploaded, the following process is performed:

```text
PDF / DOCX
    |
    v
Text Extraction
    |
    v
Section Detection
    |
    v
Section-Aware Chunking
    |
    v
Embedding Generation
    |
    v
ChromaDB
```

The document is divided into chunks while preserving section information such as:

```text
SKILLS
EXPERIENCE
EDUCATION
PROJECTS
CERTIFICATIONS
```

This section information is retained as metadata and can be used during retrieval and debugging.

---

### 2. Question Answering

When the user asks a question:

```text
User Question
      |
      v
Query Embedding
      |
      v
ChromaDB Similarity Search
      |
      v
Top-K Relevant Chunks
      |
      v
Retrieved Context
      |
      v
Groq LLM
      |
      v
Final Answer
```

The retrieved resume context and the user's question are passed to the LLM to generate a context-grounded response.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Streamlit | Web frontend |
| ChromaDB | Vector database |
| Sentence Transformers | Text embeddings |
| Groq | LLM inference |
| PyMuPDF | PDF text extraction |
| python-docx | DOCX text extraction |
| LangChain Text Splitters | Text chunking |
| python-dotenv | Environment variable management |

---

## Project Structure

```text
rag-resume-project/
|
├── app.py
|
├── src/
|   ├── __init__.py
|   ├── chunker.py
|   ├── document_loader.py
|   ├── embedding.py
|   ├── llm.py
|   ├── main.py
|   ├── retriever.py
|   └── vector_store.py
|
├── data/
|
├── chroma_db/
|
├── requirements.txt
├── .gitignore
└── README.md
```

The following files and directories are intentionally excluded from Git:

```text
.env
venv/
data/
chroma_db/
```

The `data/` directory is used for uploaded documents and should not contain personal documents in the public repository.

---

## How the Components Work

### Document Loader

The document loader handles both supported document formats:

```text
PDF  -> PyMuPDF
DOCX -> python-docx
```

The extracted content is converted into text before being passed to the chunking stage.

---

### Section-Aware Chunking

Instead of blindly splitting the complete document into fixed-size chunks, the application attempts to preserve the structure of the resume.

For example:

```text
SKILLS
    |
    +-- Programming Languages
    +-- Frameworks
    +-- Databases

PROJECTS
    |
    +-- Project 1
    +-- Project 2
    +-- Project 3

EDUCATION
    |
    +-- Degree Information

EXPERIENCE
    |
    +-- Company Information
```

Each generated chunk contains both its text and the associated section.

This allows the retrieval stage to preserve useful information about where a piece of content came from.

---

### Embedding Generation

Each document chunk is converted into a numerical vector using a Sentence Transformer embedding model.

Conceptually:

```text
Document Chunk
      |
      v
Embedding Model
      |
      v
Numerical Vector
```

The same embedding process is applied to the user's query.

```text
User Query
      |
      v
Embedding Model
      |
      v
Query Vector
```

The query vector is then compared with vectors stored in ChromaDB.

---

### ChromaDB

ChromaDB is used as the vector database.

During document ingestion:

```text
Chunk
  |
  v
Embedding
  |
  v
ChromaDB
```

During retrieval:

```text
Question
    |
    v
Query Embedding
    |
    v
ChromaDB Similarity Search
    |
    v
Relevant Chunks
```

---

### Retrieval

The retriever searches ChromaDB for chunks that are semantically similar to the user's question.

For example:

```text
Question:

"What technologies do I know?"
```

The retriever may identify chunks from:

```text
SKILLS
PROJECTS
EXPERIENCE
```

The top relevant chunks are then combined to form the context supplied to the LLM.

---

### LLM Generation

The retrieved context is passed to the Groq-hosted LLM together with the user's question.

Conceptually:

```text
Retrieved Context
       +
User Question
       |
       v
    Groq LLM
       |
       v
Generated Answer
```

The goal is to generate an answer using the information retrieved from the resume rather than relying entirely on the model's general knowledge.

---

## Example Questions

After uploading a resume, users can ask questions such as:

```text
What are my technical skills?
```

```text
What programming languages do I know?
```

```text
What projects have I worked on?
```

```text
Tell me about my work experience.
```

```text
What is my educational background?
```

```text
Which technologies have I used in my projects?
```

```text
What databases do I know?
```

---

## Retrieved Context Inspection

The application provides an option to inspect the retrieved context.

For each retrieved chunk, the application can display:

```text
Section
Similarity Distance
Chunk Content
```

For example:

```text
Context 1

Section: PROJECTS

Project information...

Distance: 0.2145
```

This is useful during development because it allows the retrieval stage to be evaluated independently from the LLM.

Instead of only checking whether the final answer is correct, the developer can verify whether the correct information was retrieved in the first place.

---

## Example RAG Flow

Suppose the user asks:

```text
What projects have I worked on?
```

The system performs:

```text
User Question
      |
      v
Query Embedding
      |
      v
ChromaDB Search
      |
      v
Relevant PROJECTS Chunks
      |
      v
Context Construction
      |
      v
Groq LLM
      |
      v
Answer
```

The retrieved context may look like:

```text
[Section: PROJECTS]

Project 1 information...

---

[Section: PROJECTS]

Project 2 information...
```

The LLM then uses this context to formulate the response.

---

## Why RAG?

A traditional approach could send the entire resume directly to an LLM:

```text
Entire Resume
      |
      v
     LLM
      |
      v
   Answer
```

The RAG approach instead retrieves only the relevant information:

```text
Entire Resume
      |
      v
   ChromaDB
      |
      v
Relevant Chunks
      |
      v
     LLM
      |
      v
   Answer
```

This approach is useful because it separates information retrieval from answer generation and can scale to larger document collections.

---

## Why ChromaDB?

ChromaDB provides vector storage and similarity search for the application.

It allows the system to store document embeddings and retrieve semantically similar chunks when a user asks a question.

The basic workflow is:

```text
Document
   |
   v
Chunks
   |
   v
Embeddings
   |
   v
ChromaDB
```

and:

```text
Question
   |
   v
Query Embedding
   |
   v
Similarity Search
   |
   v
Relevant Context
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/resume-rag-assistant.git
```

Navigate into the project:

```bash
cd resume-rag-assistant
```

---

### 2. Create a virtual environment

```bash
python -m venv venv
```

For macOS/Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## API Configuration

The application uses Groq for LLM inference.

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

If model selection is supported by the application, you can also specify:

```text
GROQ_MODEL=your_model_name
```

Do not commit `.env` to GitHub.

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

## Using the Application

### Step 1: Upload a Resume

Upload a document in one of the supported formats:

```text
PDF
DOCX
```

### Step 2: Process the Resume

Click:

```text
Process Resume
```

The application will:

```text
Extract Text
     |
     v
Create Chunks
     |
     v
Generate Embeddings
     |
     v
Store in ChromaDB
```

### Step 3: Ask Questions

Once processing is complete, enter a question in the chat interface.

For example:

```text
What are my strongest technical skills?
```

### Step 4: Inspect Retrieval

Open the retrieved-context section to check which chunks were selected for the question.

---

## Current Limitations

The current implementation is primarily designed for a single resume at a time.

Current limitations include:

- Single active resume
- Basic semantic retrieval
- Fixed top-K retrieval
- No dedicated reranking stage
- No conversation memory
- No hybrid keyword and vector search
- No automated retrieval evaluation
- Local ChromaDB storage
- Local Streamlit deployment

---

## Future Improvements

Potential improvements include:

- Hybrid search combining keyword and vector search
- Reranking retrieved chunks
- Improved section-aware chunking
- Multi-document support
- Multiple resume management
- Conversation memory
- Streaming LLM responses
- Source citations
- Retrieval confidence scoring
- Automated RAG evaluation
- Precision and recall evaluation
- Authentication
- Cloud deployment
- Persistent document management
- Query rewriting
- Metadata filtering

---

## Future Architecture

A more advanced version of the system can use a hybrid retrieval and reranking pipeline:

```text
                    User Question
                          |
                          v
                  Query Processing
                          |
                          v
              +-----------------------+
              |    Hybrid Retrieval   |
              |                       |
              | Vector Search         |
              | Keyword Search        |
              +-----------+-----------+
                          |
                          v
                  Candidate Chunks
                          |
                          v
                      Reranker
                          |
                          v
                 Top Relevant Chunks
                          |
                          v
                  Context Builder
                          |
                          v
                      Groq LLM
                          |
                          v
                    Final Answer
```

---

## Security

The following files should never be committed to a public GitHub repository:

```text
.env
data/
chroma_db/
venv/
```

The `.env` file contains API credentials.

If an API key is accidentally committed to GitHub, revoke the exposed key and generate a new one immediately.

---

## Learning Objectives

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation
- Document ingestion
- PDF and DOCX processing
- Text chunking
- Section-aware chunking
- Semantic embeddings
- Vector databases
- Similarity search
- Context retrieval
- Prompt construction
- LLM-based generation
- Streamlit application development
- Environment and API key management

---

## License

This project is intended for educational and portfolio purposes.

Add an appropriate open-source license if you decide to distribute the project publicly.

---

## Author

**Your Name**

GitHub:

```text
https://github.com/YOUR_USERNAME
```