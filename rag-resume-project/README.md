# RAG Resume Assistant

A modular, local Retrieval-Augmented Generation (RAG) pipeline designed for parsing, indexing, and querying resume documents with semantic search using **LangChain**, **ChromaDB**, **PyMuPDF**, and **Sentence-Transformers**.

---

## 📁 Project Structure

```text
rag-resume-project/
│
├── data/
│   └── resume.pdf           # Target resume PDF for indexing
│
├── chroma_db/               # Persistent ChromaDB vector database files
│
├── src/
│   ├── __init__.py
│   ├── loader.py            # PDF & text extraction using PyMuPDF (fitz)
│   ├── chunker.py           # Recursive text chunking with metadata enrichment
│   ├── embeddings.py        # Sentence-Transformers (all-MiniLM-L6-v2) embedding model
│   ├── vectorstore.py       # ChromaDB persistent store management
│   ├── retrieve.py          # Semantic similarity search & result formatting
│   └── main.py              # CLI entry point for indexing and querying
│
├── requirements.txt         # Project dependencies
└── README.md                # Documentation and usage guide
```

---

## 🚀 Getting Started

### 1. Activate your virtual environment

```bash
cd rag-resume-project
source venv/bin/activate
```

### 2. Install dependencies (if not already installed)

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Build or Rebuild the Vector Index
Index `data/resume.pdf` into `chroma_db/`:
```bash
python src/main.py --index
```

You can also specify a custom resume path:
```bash
python src/main.py --index --file path/to/your_resume.pdf
```

### 2. Single Question / Query
Retrieve the top matching chunks for a question directly from the terminal:
```bash
python src/main.py --query "What is the candidate's experience with Python and FastAPI?"
```

Specify number of results (`--top-k`):
```bash
python src/main.py --query "What degrees and university did the candidate attend?" --top-k 2
```

### 3. Interactive Q&A Mode
Run interactive question-answering mode:
```bash
python src/main.py --interactive
```
Or simply:
```bash
python src/main.py
```

---

## 🧩 Module Breakdown

| Module | Responsibility |
| :--- | :--- |
| [`src/loader.py`](src/loader.py) | Extracts text and metadata (page numbers, filenames) from PDF or text resumes. |
| [`src/chunker.py`](src/chunker.py) | Splits text into overlapping chunks using `RecursiveCharacterTextSplitter`. |
| [`src/embeddings.py`](src/embeddings.py) | Generates dense embeddings locally using `all-MiniLM-L6-v2` (MPS/CUDA/CPU). |
| [`src/vectorstore.py`](src/vectorstore.py) | Creates and loads persistent ChromaDB vector collections. |
| [`src/retrieve.py`](src/retrieve.py) | Executes cosine/L2 semantic search and formats results with citations. |
| [`src/main.py`](src/main.py) | Ties all components together into an easy-to-use CLI. |
