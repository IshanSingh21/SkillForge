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
| 🎯 Skill Extraction | 🔲 M3 | Hybrid NLP + embedding-based skill identification |
| 🔗 Semantic Matching | 🔲 M3 | Embedding-based resume ↔ job description matching |
| 📊 Match Score | 🔲 M3 | Weighted semantic similarity score |
| 🗺️ Learning Roadmap | 🔲 M5 | LLM-generated personalized skill development plan |
| 🎤 Interview Questions | 🔲 M5 | Role-specific, gap-aware interview preparation |
| 💬 RAG Career Assistant | 🔲 M4 | Ask questions grounded in your resume & career KB |
| 📌 Source Citations | 🔲 M4 | Every answer cites its source documents |

> 🟢 = Implemented · 🔲 = Planned

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                       │
├─────────────────────────────────────────────────────────────┤
│                     Service Layer                           │
│  ResumeService · SkillExtractor · SemanticMatcher           │
│  RoadmapGenerator · InterviewGenerator · RAGAssistant       │
├─────────────────────────────────────────────────────────────┤
│                    AI / ML Core                             │
│  EmbeddingEngine (SentenceTransformers)                     │
│  VectorStore (FAISS)  ·  LLMProvider (Gemini / Groq)       │
├─────────────────────────────────────────────────────────────┤
│                     Data Layer                              │
│  PDFParser (PyMuPDF) · Preprocessor · Chunker               │
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

# 6. Run the application
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

```
skillforge-ai/
├── config/settings.py           # Pydantic Settings (loads .env)
├── src/skillforge/
│   ├── data/                    # PDF parsing, text cleaning, chunking
│   ├── ai/                     # Embeddings, vector store, LLM providers
│   ├── services/               # Business logic orchestration
│   ├── models/                 # Pydantic data models
│   └── utils/                  # Logging, exceptions
├── knowledge_base/             # Curated career knowledge documents
├── app/                        # Streamlit frontend
│   ├── streamlit_app.py        # Main entry point
│   ├── pages/                  # Multi-page app pages
│   └── components/             # Reusable UI components
├── tests/                      # pytest test suite
└── scripts/                    # Utility scripts
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/skillforge --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_pdf_parser.py -v
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Interactive web UI |
| NLP | spaCy, SentenceTransformers | Text processing & embeddings |
| Vector Search | FAISS | Similarity search |
| LLM | Gemini / Groq (swappable) | Text generation |
| Data Models | Pydantic v2 | Validation & serialization |
| PDF Parsing | PyMuPDF | Resume text extraction |
| Config | pydantic-settings, python-dotenv | Environment management |
| Logging | Loguru | Structured logging |
| Testing | pytest | Unit & integration tests |

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🗺️ Roadmap

- [x] **Milestone 1**: Foundation & Data Layer
- [ ] **Milestone 2**: AI Core (Embeddings, Vector Store, LLM Abstraction)
- [ ] **Milestone 3**: Skill Extraction & Semantic Matching
- [ ] **Milestone 4**: RAG Pipeline & Career Assistant
- [ ] **Milestone 5**: Generation Services (Roadmap & Interview)
- [ ] **Milestone 6**: Streamlit UI
- [ ] **Milestone 7**: Testing, Polish & Deployment
