# 📄 CV RAG System

An intelligent, HR-focused **Retrieval-Augmented Generation (RAG)** system designed to chat with multiple CVs simultaneously. This tool uses LLM-driven structural chunking to parse resumes more accurately than traditional fixed-size window methods.

## 🚀 Features

* **LLM-Driven Structural Chunking**: Instead of arbitrary character limits, the system uses `Llama 3.3` to logically identify CV sections (e.g., Experience, Education, Skills).
* **Multi-CV Context**: Upload multiple PDFs and query across all of them or filter for specific candidates automatically.
* **Vector Search with Qdrant**: High-performance vector similarity search using `all-MiniLM-L6-v2` embeddings.
* **Session Isolation**: Securely tracks which CVs are active in your current session using file hashing to prevent redundant indexing.
* **HR-Optimized Responses**: Answers are strictly grounded in the provided CV excerpts with clear candidate attribution.

---

## 🛠️ Tech Stack

**Frontend** | [Streamlit](https://streamlit.io/)

**Orchestration** | Python, [Groq SDK](https://github.com/groq/groq-python) 

**Vector Database** | [Qdrant](https://qdrant.tech/) 

**Embeddings** | `sentence-transformers/all-MiniLM-L6-v2`

**LLM** | `Llama-3.3-70b-versatile` (via Groq)

**PDF Parsing** | `pypdf`

---

## 📋 Prerequisites

* Python 3.10+
* A **Groq Cloud** API Key
* A **Qdrant** instance (Cloud or Local Docker)

## ⚙️ Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/cv-rag-system.git
cd cv-rag-system

```


2. **Install dependencies:**
```bash
pip install -r req.txt

```


3. **Configure Environment Variables:**
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_key_here
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=cv_collection
GROQ_MODEL=llama-3.3-70b-versatile

```


4. **Initialize the Database:**
Run the setup script to create the Qdrant collection and payload indexes:
```bash
python qdrant_setup.py

```



---

## 🖥️ Usage

1. **Start the application:**
```bash
streamlit run app.py

```


2. 
**Upload CVs**: Drag and drop PDF resumes into the sidebar/upload section.


3. **Search**: Ask questions like:
* *"Which candidates have experience with Kubernetes?"*
* *"Compare the educational backgrounds of Saif and Jane."*
* *"Summarize the work history for the candidate from Google."*



## 🧠 How it Works

1. **Extraction**: `pypdf` extracts raw text from the upload.
2. **Chunking**: The LLM analyzes the text to identify logical sections (Summary, Skills, etc.), ensuring context remains intact.
3. **Indexing**: Chunks are converted to vectors and stored in **Qdrant** with metadata (file hash, candidate name, section type).
4. **Retrieval**: When you ask a question, the system filters vectors by your session's file hashes and performs a similarity search.
5. **Generation**: The retrieved excerpts are fed to Llama 3.3 to generate a structured, factual HR response.
