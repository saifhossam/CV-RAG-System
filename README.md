# 👔 TalentTrace RAG

**TalentTrace RAG** is an intelligent HR recruitment assistant designed to transform static CVs into an interactive knowledge base. By utilizing **LLM-driven structural chunking**, it moves beyond simple keyword matching to understand the logical sections of a resume—allowing recruiters to query specific candidates or skills with unprecedented precision.

---

## 🌟 Core HR Features

* **Structural Intelligence**
  Uses `Llama-3.3-70b` to logically split CVs into sections such as *Work Experience*, *Education*, and *Technical Skills*, rather than arbitrary text blocks.

* **Fact-Grounded Responses**
  The system acts as an *Expert HR Assistant*, answering questions using **only retrieved CV excerpts** to prevent hallucinations.

* **Automatic Candidate Attribution**
  Every generated response is explicitly attributed to the relevant candidate by name.

* **Session-Based Isolation**
  Multiple candidates can be processed within a single session, with optional filtering by candidate name or file hash.

* **Source Transparency**
  Includes a *View Retrieved Sources* expander, enabling HR professionals to verify the raw resume content behind every AI-generated claim.

---

## 🏗️ Database Architecture

The system relies on **Qdrant** for high-performance vector retrieval. The setup script initializes the environment with production-oriented optimizations:

### Vector Configuration

* **Vector Size:** 384
* **Embedding Model:** `all-MiniLM-L6-v2`
* **Similarity Metric:** Cosine Distance

### Payload Indexing

To ensure low-latency filtering at scale, the following fields are indexed as `KEYWORD` types:

* **`file_hash`**
  Prevents duplicate indexing and enforces session-level isolation.
* **`candidate_name` / `candidate_name_lower`**
  Enables instant filtering when a recruiter mentions a candidate by name.
* **`section`**
  Allows retrieval to prioritize or isolate specific resume sections.

---

## 🛠️ Technical Stack

| Component         | Technology                                 |
| ----------------- | ------------------------------------------ |
| **Interface**     | Streamlit                                  |
| **LLM Inference** | Groq (Llama 3.3 70B)                       |
| **Vector Store**  | Qdrant                                     |
| **Embeddings**    | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **PDF Engine**    | PyPDF                                      |

---

## 🚀 Quick Start

### 1. Configure Environment

Create a `.env` file with the following variables:

```env
GROQ_API_KEY=your_key
QDRANT_URL=your_url
QDRANT_API_KEY=your_qdrant_key
COLLECTION_NAME=cv_collection
```

### 2. Initialize Database

Run the setup script **once** to create the collection and payload indexes:

```bash
python qdrant_setup.py
```

---

## 🖥️ Usage

### Start the Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

### Upload CVs

Upload one or more PDF resumes using the sidebar upload component.
Each CV is automatically processed and isolated within the current session to maintain candidate-level accuracy.

### Search & Query

Ask natural-language questions such as:

* *Which candidates have experience with Kubernetes?*
* *Compare the educational backgrounds of Saif and Jane.*
* *Summarize the work history of the candidate who worked at Google.*

The system automatically identifies relevant candidates, retrieves supporting evidence, and generates a structured HR-focused response.

---

## 🧠 How It Works

TalentTrace follows a deterministic and auditable pipeline to ensure factual accuracy and transparency:

### 1. Text Extraction

Uploaded PDF resumes are processed using **PyPDF**, extracting raw textual content while preserving section order.

### 2. Structural Chunking

A large language model analyzes the extracted text and identifies logical resume sections (e.g., *Summary*, *Skills*, *Work Experience*, *Education*).
This preserves contextual integrity and prevents unrelated information from being mixed.

### 3. Vector Indexing

Each structured chunk is embedded and stored in **Qdrant**, along with metadata such as:

* `file_hash`
* `candidate_name`
* `section`

### 4. Filtered Retrieval

When a query is submitted, the system:

* Restricts retrieval to the current session’s `file_hashes`
* Applies candidate-name filters when detected
* Performs a similarity search to identify the most relevant resume sections

### 5. Answer Generation

The retrieved excerpts are passed to **Llama 3.3 (70B)** with strict constraints:

* Responses must rely **only** on retrieved content
* Output is structured, concise, and candidate-attributed
* No external knowledge or assumptions are introduced
