# 🧠 SkillForge AI

**Intelligent Resume Analysis & Career Development Platform**

SkillForge AI is a portfolio-grade AI/ML application that uses NLP, sentence-transformer embeddings, semantic search, RAG (Retrieval-Augmented Generation), and LLMs to help professionals analyze their resumes against job descriptions, identify skill gaps, and build personalized career development plans.

---

## ✨ Features

| Feature | Status | Description |
|---|---|---|
| 📄 Resume PDF Upload | 🟢 M1 | Upload and extract text from PDF resumes |
| 🧹 Text Preprocessing | 🟢 M1 | Clean, normalize, and section-parse resume text |
| ✂️ Document Chunking | 🟢 M1 | Recursive text splitting with configurable overlap |
| 🎯 Skill Extraction | 🟢 M3 | Hybrid NLP + pattern-based skill identification with taxonomy |
| 🧬 Semantic Embeddings | 🟢 M4 | Sentence-Transformers (`all-MiniLM-L6-v2`) with cosine similarity |
| 📊 Explainable Matching | 🟢 M5 | Multi-factor transparent scoring (skills, coverage, semantics, experience) |
| 🗄️ FAISS Vector Retrieval | 🟢 M6 | Multi-namespace vector store for Resume, JD, and Knowledge Base |
| 💬 Grounded RAG Pipeline | 🟢 M7 | Source-grounded conversational assistant with anti-hallucination guardrails |
| 📌 Source Citations | 🟢 M7 | Every answer cites its source documents and relevance scores |
| 🗺️ Learning Roadmap | 🔲 M8 | LLM-generated personalized skill development plan |
| 🎤 Interview Questions | 🔲 M8 | Role-specific, gap-aware interview preparation |

> 🟢 = Implemented · 🔲 = Planned

---

## 💬 Grounded RAG Pipeline (Milestone 7)

SkillForge AI implements an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline designed for factuality, provenance, and zero hallucinations.

```
 User Query
     │
     ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  1. Query Embedding (all-MiniLM-L6-v2, 384 dimensions)      │
 └─────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  2. FAISS Vector Retrieval (Resume, JD, Knowledge Base)    │
 │     Filtered by cosine similarity threshold & top-k         │
 └─────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  3. Context Formatting & Provenance Metadata Assembly       │
 │     [Document 1: Resume - Section: Experience | Rel: 0.85]  │
 └─────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  4. Prompt Construction with Strict Anti-Hallucination      │
 │     System Directives                                       │
 └─────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  5. Pluggable LLM Backend (Google Gemini / Groq / Mock)     │
 └─────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  6. Grounded Response with Typed SourceCitation Objects     │
 └─────────────────────────────────────────────────────────────┘
```

### 🛡️ Anti-Hallucination Guardrails

1. **Strict Context Grounding**: The LLM is given strict system directives to base answers **exclusively** on the provided retrieved context.
2. **Explicit Fallback on Missing Data**: If the retrieved documents lack sufficient facts, the assistant explicitly states: *"Based on the provided documents, I do not have enough information to answer this question."*
3. **Traceable Citations**: Every response returns a structured `list[SourceCitation]` containing:
   - Source Type (`resume`, `job_description`, `knowledge_base`)
   - Document & Section Name
   - Content Preview snippet
   - Cosine Relevance Score (0.0 to 1.0)
4. **Pluggable LLM Provider Contract**: The RAG pipeline relies on an abstract `LLMProvider` interface, allowing hot-swapping between Google Gemini, Groq, Ollama, or deterministic Mock providers without altering retrieval code.

---

## 🗄️ Vector Retrieval & Chunking Infrastructure (Milestone 6)

SkillForge AI features a multi-source vector retrieval architecture powered by **FAISS (`faiss-cpu`)** and dense sentence embeddings.

### ✂️ Chunk Size & Overlap Strategy

- **Chunk Size: 512 characters (~80–100 words / 1 paragraph)**
  - Matches the 256-token receptive field of `all-MiniLM-L6-v2`.
  - Encapsulates complete semantic units (an achievement bullet point or job requirement clause).
- **Chunk Overlap: 50 characters (~8–10 words)**
  - Preserves technical multi-word phrases across boundary splits.

### 📚 Isolated Vector Partitions (Namespaces)

1. **`resume`**: Candidate work experience, education, projects, and skills.
2. **`job_description`**: Target role requirements, qualifications, and responsibilities.
3. **`knowledge_base`**: Curated career transition guides, interview strategies, and market trends.

---

## 📊 Resume-Job Matching & Scoring Methodology

$$\text{Overall Score} = 0.40 \cdot S_{\text{skills}} + 0.25 \cdot S_{\text{req}} + 0.20 \cdot S_{\text{semantic}} + 0.15 \cdot S_{\text{exp}}$$

| Factor | Weight | Formula / Definition | Purpose |
|---|---|---|---|
| **1. Skill Match Quality ($S_{\text{skills}}$)** | **40%** | $\frac{\sum w_i \cdot \text{sim}_i}{\sum w_i} \times 100$ | Weighted average of semantic & exact skill matches. Required skills carry $1.5\times$ weight over preferred skills. |
| **2. Required Skills Coverage ($S_{\text{req}}$)** | **25%** | $\frac{\text{matched}_{\text{req}} + 0.5 \cdot \text{partial}_{\text{req}}}{\text{total}_{\text{req}}} \times 100$ | Proportion of mandatory/essential job requirements satisfied by the candidate. |
| **3. Content Semantic Fit ($S_{\text{semantic}}$)** | **20%** | $\min\left(100, \max\left(0, \frac{\text{cosine\_sim} - 0.20}{0.80 - 0.20} \times 100\right)\right)$ | Dense sentence-transformer embedding cosine similarity between the full resume and job description. |
| **4. Experience & Seniority ($S_{\text{exp}}$)** | **15%** | $\min\left(100, \frac{\text{Years}_{\text{candidate}}}{\text{Years}_{\text{required}}} \times 100\right) \pm \text{Seniority Adjustment}$ | Evaluates minimum years of experience and seniority level alignment (Junior $\to$ Senior $\to$ Lead). |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                       │
│  01_Upload · 02_Analysis · 03_Roadmap · 04_Interview · 05_Assistant
├─────────────────────────────────────────────────────────────┤
│                     Service Layer                           │
│  ResumeService · SkillExtractor · SemanticMatcher           │
│  RetrievalService · RAGAssistant · RoadmapGenerator         │
├─────────────────────────────────────────────────────────────┤
│                    AI / ML Core                             │
│  EmbeddingEngine (SentenceTransformers all-MiniLM-L6-v2)    │
│  VectorStore (FAISS IndexFlatIP) · LLMProvider (Gemini/Groq/Mock)
├─────────────────────────────────────────────────────────────┤
│                     Data Layer                              │
│  PDFParser (PyMuPDF) · Preprocessor · Chunker · Taxonomy   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A free [Google Gemini API key](https://aistudio.google.com/apikey) or [Groq API key](https://console.groq.com/keys)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/skillforge-ai.git
cd skillforge-ai

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Configure environment variables
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
# Edit .env and add your API keys

# 6. Index knowledge base documents into FAISS
python scripts/index_knowledge_base.py

# 7. Run the application
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

```
skillforge-ai/
├── config/settings.py           # Pydantic Settings (loads .env)
├── src/skillforge/
│   ├── data/                    # PDF parsing, text cleaning, chunking, skill taxonomy
│   ├── ai/
│   │   ├── embeddings.py        # SentenceTransformers engine
│   │   ├── vector_store.py      # FAISS multi-namespace vector database
│   │   └── llm/                 # Abstract LLMProvider, Gemini, Groq, Mock & Factory
│   ├── services/
│   │   ├── resume_service.py    # PDF processing & extraction pipeline
│   │   ├── skill_extractor.py   # Hybrid taxonomy & spaCy NLP skill extractor
│   │   ├── semantic_matcher.py  # Multi-factor explainable scoring engine
│   │   └── rag_assistant.py     # End-to-end grounded RAG pipeline
│   ├── models/                  # Pydantic models (matching, resume, roadmap, rag)
│   └── utils/                   # Logging, custom exception hierarchy
├── knowledge_base/             # Curated career guides (interview, trends, transitions)
├── app/                        # Streamlit frontend (Upload, Analysis, Assistant)
├── tests/                      # pytest test suite (175 tests)
└── scripts/                    # index_knowledge_base.py, evaluate_matching.py
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/skillforge --cov-report=term-missing

# Run RAG pipeline tests
pytest tests/unit/test_rag_pipeline.py -v
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Interactive web UI & analytics dashboard |
| NLP & Embeddings | spaCy, SentenceTransformers (`all-MiniLM-L6-v2`) | Skill extraction, dense embeddings & semantic matching |
| Vector Search | FAISS (`faiss-cpu`) | Partitioned similarity search across Resume, JD & Knowledge Base |
| LLM | Gemini / Groq / Mock (swappable) | Contextual generation with strict anti-hallucination grounding |
| Data Models | Pydantic v2 | Type validation & serialization |
| PDF Parsing | PyMuPDF | Structured resume extraction |
| Config | pydantic-settings, python-dotenv | Environment management |
| Logging | Loguru | Structured logging |
| Testing | pytest | Comprehensive unit & integration testing (175 tests) |

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
