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
| 🗺️ Personalized Roadmap | 🟢 M9 | Explainable, 3-stage learning plan grounded in career knowledge |
| 🎤 Interview Preparation | 🟢 M10 | Personalized technical, conceptual, project, and behavioral questions |

> 🟢 = Implemented & Verified (211 Automated Tests)

---

## 🎤 Personalized Interview Preparation (Milestone 10)

SkillForge AI generates role-tailored, gap-aware interview preparation question sets dynamically calibrated to the candidate's seniority and resume background.

### 🗂️ 4 Balanced Question Categories

| Category | Purpose | Grounding Source |
|---|---|---|
| 💻 **Technical** | Probes hands-on API usage, framework mechanics, and identified **missing skill gaps**. | Grounded in FAISS Career Knowledge Base with citations |
| 🧠 **Conceptual** | Tests theoretical foundations, algorithmic trade-offs, and computational complexity. | Grounded in domain theory guides with citations |
| 🛠️ **Project-Based** | Deep-dives into actual projects, metrics, and architecture parsed from candidate resume. | Extracted directly from `ResumeAnalysis` sections |
| 🤝 **Behavioral** | Evaluates incident response, technical conflict resolution, and leadership via STAR method. | Calibrated to candidate experience level |

---

## 🗺️ Personalized Learning Roadmap (Milestone 9)

SkillForge AI generates an explainable, structured **3-stage learning roadmap** based on skill gaps from Milestone 5 and grounded in the career knowledge base from Milestone 8.

```
 Candidate Skill Gaps (M5) ──┐
                             ├─► Prerequisite DAG ──► Multi-Factor Prioritization ──► 3 Sequential Stages ──► Grounded Roadmap
 Career Knowledge Base (M8) ──┘
```

### 📐 Multi-Factor Prioritization Formula

$$\text{Priority Score} = \text{Requirement Base} + \text{Gap Status} + \text{Prerequisite Dependency Boost} + \text{Knowledge Grounding Boost}$$

- **Requirement Base**: Required ($+50$) vs Preferred ($+20$)
- **Gap Status**: Missing ($+30$) vs Partial Match ($+15$)
- **Prerequisite Dependency**: $+15$ boost if another target skill depends on this skill
- **Knowledge Base Grounding**: $+5$ boost if backed by retrieved domain documents

### 🪜 3 Sequential Learning Stages

1. **Stage 1: Foundations & Critical Prerequisites**: Urgent foundational gaps (e.g. Python, SQL, Math) and prerequisite dependencies needed for subsequent learning.
2. **Stage 2: Core Role Competencies**: Primary required frameworks, libraries, and tools for the target role whose prerequisites are met.
3. **Stage 3: Advanced Specialization & Production**: Preferred / nice-to-have tools, MLOps, CI/CD, cloud deployments, and capstone portfolio projects.

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

---

## 💬 Grounded RAG Pipeline (Milestone 7)

```
 User Query ──► Query Embedding ──► FAISS Retrieval ──► Context Assembly ──► Grounded Prompt ──► LLM ──► Grounded Response + Citations
```

### 🛡️ Anti-Hallucination Guardrails

1. **Strict Context Grounding**: The LLM is given strict system directives to base answers **exclusively** on the provided retrieved context.
2. **Explicit Fallback on Missing Data**: If the retrieved documents lack sufficient facts, the assistant explicitly states: *"Based on the provided documents, I do not have enough information to answer this question."*
3. **Traceable Citations**: Every response returns a structured `list[SourceCitation]` containing source document, section, and relevance score.
4. **Pluggable LLM Provider Contract**: Decoupled abstract `LLMProvider` interface (Gemini, Groq, Mock, Ollama).

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
# Run all tests (211 unit and integration tests)
pytest

# Run with coverage
pytest --cov=src/skillforge --cov-report=term-missing

# Run interview generator tests
pytest tests/unit/test_interview_generator.py -v
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Interactive web UI, roadmap stages, interview prep & analytics dashboard |
| NLP & Embeddings | spaCy, SentenceTransformers (`all-MiniLM-L6-v2`) | Skill extraction, dense embeddings & semantic matching |
| Vector Search | FAISS (`faiss-cpu`) | Partitioned similarity search across Resume, JD & Knowledge Base |
| LLM | Gemini / Groq / Mock (swappable) | Contextual generation with strict anti-hallucination grounding |
| Data Layer | PyMuPDF, `KnowledgeBaseLoader` | Multi-source parsing and structured markdown ingestion |
| Testing | pytest | Comprehensive test suite (211 tests) |
