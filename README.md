# 📄 CV Intelligence System

A Retrieval-Augmented Generation (RAG) application for querying and evaluating CVs using hybrid vector search and Azure OpenAI. Upload multiple CVs, ask natural language questions across them, and generate structured strength/weakness reports against a job description.

---

## ✨ Features

- **Multi-CV Upload & Indexing** — Upload one or more PDF CVs; each is parsed, chunked by section, and stored with deduplication via file hashing.
- **Hybrid Search (Dense + Sparse)** — Combines Azure OpenAI embeddings (dense) with SPLADE sparse embeddings, merged via Reciprocal Rank Fusion (RRF) for high-quality retrieval.
- **Conversational Q&A** — Ask questions across all uploaded CVs; the system filters by candidate name when mentioned in the query.
- **CV Strength & Weakness Report** — Upload a single CV alongside a job description to get a structured evaluation: match summary, strengths, gaps, missing skills, and a match score.
- **Persistent Vector Store** — CVs are stored in Qdrant; re-uploading a previously indexed CV skips re-processing automatically.

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
extract_text_from_pdf()         # pypdf
    │
    ▼
llm_structural_chunking()       # Azure OpenAI → section-aware chunks
    │
    ▼
index_documents()
    ├── _dense_embed()           # text-embedding-3-small (Azure OpenAI)
    └── _sparse_embed()          # SPLADE (fastembed)
          │
          ▼
       Qdrant (named vectors: "dense" + "sparse")

Query
    │
    ▼
retrieve()
    ├── Dense search
    ├── Sparse search
    └── RRF merge
          │
          ▼
generate_answer() / generate_cv_strength_report()   # Azure OpenAI GPT
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM & Embeddings | Azure OpenAI (GPT + text-embedding-3-small) |
| Sparse Embeddings | fastembed + SPLADE (`prithivida/Splade_PP_en_v1`) |
| Vector Database | Qdrant |
| PDF Parsing | pypdf |
| Config | python-dotenv |

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- A running [Qdrant](https://qdrant.tech/) instance (cloud or self-hosted)
- Azure OpenAI resource with:
  - A **chat** deployment (e.g. `gpt-4.1-nano` or `gpt-4o`)
  - An **embedding** deployment (`text-embedding-3-small`)

### 2. Clone the repository

```bash
git clone https://github.com/your-username/cv-intelligence-system.git
cd cv-intelligence-system
```

### 3. Install dependencies

```bash
pip install -r req.txt
pip install fastembed openai
```

> **Note:** `fastembed` and `openai` are runtime dependencies not yet listed in `req.txt`. Add them if you update the file.

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-nano          # Chat model deployment name
AZURE_CHAT_DEPLOYMENT=gpt-4.1-nano            # Used by loader.py for chunking
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=cv_collection                  # Optional, defaults to cv_collection
```

### 5. Initialize the Qdrant collection

Run this once to create the collection and payload indexes:

```bash
python qdrant_setup.py
```

### 6. Launch the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py              # Streamlit UI — two tabs: Chat and Report
├── loader.py           # PDF extraction and LLM-based section chunking
├── rag.py              # Embeddings, vector store ops, retrieval, generation
├── qdrant_setup.py     # One-time Qdrant collection initialisation
├── req.txt             # Python dependencies
└── .env                # Environment variables (not committed)
```

---

## 🖥️ Usage

### Tab 1 — Chat with CVs

1. Upload one or more CV PDFs using the file uploader.
2. Each CV is parsed into sections (Education, Work Experience, Skills, etc.) and indexed.
3. Type a natural language question (e.g. *"Who has Python experience?"*, *"What is Ahmed's education background?"*).
4. The system retrieves the most relevant sections and generates a cited answer.

### Tab 2 — CV Strength & Weakness Report

1. Upload a single CV PDF.
2. Paste the full job description into the text area.
3. Click **Analyze CV** to receive a structured report covering:
   - Overall match summary
   - Key strengths
   - Weaknesses and gaps
   - Missing skills
   - Estimated match score (0–100%)

---

## ⚙️ Configuration Reference

| Variable | Description | Default |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | 
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | 
| `AZURE_OPENAI_API_VERSION` | API version | `2024-12-01-preview` |
| `AZURE_OPENAI_DEPLOYMENT` | Chat model deployment name | `gpt-4.1-nano` |
| `AZURE_CHAT_DEPLOYMENT` | Chat deployment used by loader |
| `AZURE_OPENAI_EMBEDDING_MODEL` | Embedding model deployment name | `text-embedding-3-small` |
| `QDRANT_URL` | Qdrant instance URL |
| `QDRANT_API_KEY` | Qdrant API key | 
| `COLLECTION_NAME` | Qdrant collection name |

---

## 🔍 How Retrieval Works

1. The query is embedded using both **dense** (Azure OpenAI) and **sparse** (SPLADE) models.
2. Both embeddings are searched in parallel against the Qdrant collection, filtered to only the CVs uploaded in the current session.
3. If a candidate name is detected in the query, an additional payload filter narrows results to that candidate.
4. Results from both searches are merged using **Reciprocal Rank Fusion (RRF)**, which balances lexical and semantic signals without requiring score normalisation.
5. The top-ranked chunks are passed to GPT as context for answer generation.

---

## 📝 Notes

- **Deduplication:** CVs are hashed (SHA-256) on upload. Re-uploading the same file skips re-indexing and reuses existing vectors.
- **Fallback chunking:** If the LLM fails to parse sections, the entire CV text is stored as a single chunk.
- **Content policy:** Azure content filter errors are caught and surfaced as user-friendly messages.
- **Language:** The answer generation prompt instructs the model to reply in the same language as the question.
