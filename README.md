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
| 📖 Curated Knowledge Base | 🟢 M8 | Local structured knowledge base across 10 core technical domains |
| 🗺️ Learning Roadmap | 🔲 M9 | LLM-generated personalized skill development plan |
| 🎤 Interview Questions | 🔲 M9 | Role-specific, gap-aware interview preparation |

> 🟢 = Implemented · 🔲 = Planned

---

## 📖 Curated Career Knowledge Base (Milestone 8)

SkillForge AI includes a local, curated knowledge base structured across 10 core technology and computer science domains:

| Domain / Topic | Document | Covered Sections |
|---|---|---|
| **Machine Learning** | `machine_learning.md` | Supervised/unsupervised theory, Scikit-Learn, evaluation metrics, feature engineering, learning progression |
| **Deep Learning** | `deep_learning.md` | PyTorch, backpropagation, CNNs, Transformers, loss functions, GPU acceleration, career roles |
| **Natural Language Processing** | `natural_language_processing.md` | spaCy, tokenization, embeddings, Transformers (BERT/RoBERTa), vector search, project ideas |
| **Computer Vision** | `computer_vision.md` | OpenCV, YOLO object detection, segmentation (U-Net), Vision Transformers, edge deployment |
| **SQL & Databases** | `sql_and_databases.md` | CTEs, window functions, schema design, ACID transactions, indexing, query optimization, dbt |
| **Python Development** | `python_development.md` | OOP, dunder methods, type hinting, `asyncio`, FastAPI microservices, testing with pytest |
| **MLOps** | `mlops.md` | CI/CD for ML, MLflow tracking, data/concept drift monitoring, model registries, serving (vLLM/Triton) |
| **Cloud Computing** | `cloud_computing.md` | AWS/GCP, VPC networking, Terraform IaC, IAM security, serverless Lambda, Docker/Kubernetes |
| **Data Structures & Algorithms** | `data_structures_and_algorithms.md` | Complexity analysis, trees, heaps, graphs, dynamic programming, technical interview patterns |
| **Generative AI** | `generative_ai.md` | Autoregressive LLMs, RAG architecture, LoRA/PEFT fine-tuning, guardrails, AI agent workflows |

### 🔄 Ingestion & Retrieval Pipeline

```
 Local Markdown Files (*.md)
              │
              ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  1. KnowledgeBaseLoader                                      │
 │     • Parses H1 titles and H2 section hierarchies           │
 │     • Extracts topic, doc_id, file path, and search tags    │
 └─────────────────────────────────────────────────────────────┘
              │
              ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  2. TextChunker (512 chars max, 50 chars overlap)           │
 │     • Preserves section headings & topic tags per chunk     │
 └─────────────────────────────────────────────────────────────┘
              │
              ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  3. EmbeddingEngine (all-MiniLM-L6-v2, 384 dimensions)      │
 │     • Computes dense L2-normalized vector embeddings        │
 └─────────────────────────────────────────────────────────────┘
              │
              ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  4. FAISS VectorStore ('knowledge_base' partition)          │
 │     • IndexFlatIP exact cosine search with metadata filter   │
 └─────────────────────────────────────────────────────────────┘
              │
              ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  5. RAGAssistant                                            │
 │     • recommend_career_path() & recommend_progression()     │
 └─────────────────────────────────────────────────────────────┘
```

---

## 💬 Grounded RAG Pipeline (Milestone 7)

SkillForge AI implements an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline designed for factuality, provenance, and zero hallucinations.

```
 User Query ──► Query Embedding ──► FAISS Retrieval ──► Context Assembly ──► Grounded Prompt ──► LLM ──► Grounded Response + Citations
```

### 🛡️ Anti-Hallucination Guardrails

1. **Strict Context Grounding**: The LLM is given strict system directives to base answers **exclusively** on the provided retrieved context.
2. **Explicit Fallback on Missing Data**: If the retrieved documents lack sufficient facts, the assistant explicitly states: *"Based on the provided documents, I do not have enough information to answer this question."*
3. **Traceable Citations**: Every response returns a structured `list[SourceCitation]` containing:
   - Source Type (`resume`, `job_description`, `knowledge_base`)
   - Document & Section Name
   - Content Preview snippet
   - Cosine Relevance Score (0.0 to 1.0)
4. **Pluggable LLM Provider Contract**: The RAG pipeline relies on an abstract `LLMProvider` interface, allowing hot-swapping between Google Gemini, Groq, Ollama, or deterministic Mock providers.

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

## 🚀 Quick Start

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

# 5. Index knowledge base documents into FAISS
python scripts/index_knowledge_base.py

# 6. Run the application
streamlit run app/streamlit_app.py
```

---

## 🧪 Running Tests

```bash
# Run all tests (195 unit and integration tests)
pytest

# Run with coverage
pytest --cov=src/skillforge --cov-report=term-missing

# Run knowledge base tests
pytest tests/unit/test_knowledge_base.py -v
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Interactive web UI & analytics dashboard |
| NLP & Embeddings | spaCy, SentenceTransformers (`all-MiniLM-L6-v2`) | Skill extraction, dense embeddings & semantic matching |
| Vector Search | FAISS (`faiss-cpu`) | Partitioned similarity search across Resume, JD & Knowledge Base |
| LLM | Gemini / Groq / Mock (swappable) | Contextual generation with strict anti-hallucination grounding |
| Data Layer | PyMuPDF, `KnowledgeBaseLoader` | Multi-source parsing and structured markdown ingestion |
| Testing | pytest | Comprehensive test suite (195 tests) |
