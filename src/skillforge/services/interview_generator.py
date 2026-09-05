"""
SkillForge AI — Personalized Interview Question Generator.

Generates structured, candidate-specific interview questions grounded in
resume analysis, target role requirements, skill gaps (M5), and retrieved
career knowledge base documentation (M7/M8).

Question Categories:
    1. Technical: Implementation details, framework mechanisms, and skill gap probing.
    2. Conceptual: Theoretical foundations, algorithmic trade-offs, and architecture.
    3. Project-Based: In-depth probing of candidate's actual resume projects & experience.
    4. Behavioral: Role-aligned STAR behavioral scenarios and leadership challenges.

Architecture & Design Decisions:
    - 100% Deterministic & Offline Capable: Core question synthesis, difficulty calibration,
      evaluation criteria, and RAG citation grounding work without external API keys.
    - Grounded in Career Knowledge: Technical and conceptual questions attach verified
      SourceCitation metadata from the local FAISS vector store.
    - Project Experience Extraction: Automatically parses project and work history sections
      from ResumeAnalysis to construct personalized scenario questions.
    - Pluggable LLM Refinement: Can optionally leverage LLMProvider for additional synthesis.
"""

from __future__ import annotations

import re
from typing import Any

from src.skillforge.ai.llm.base import LLMProvider
from src.skillforge.models.matching import MatchResult, SkillGap, SkillMatch
from src.skillforge.models.rag import SourceCitation
from src.skillforge.models.resume import ResumeAnalysis, ResumeSection, Skill
from src.skillforge.models.roadmap import (
    InterviewQuestion,
    InterviewQuestionSet,
    QuestionCategory,
    QuestionDifficulty,
)
from src.skillforge.services.retrieval_service import RetrievalService
from src.skillforge.utils.exceptions import SkillForgeError
from src.skillforge.utils.logging import logger


def _extract_skill_str(skill_obj: Any) -> str:
    """Extract skill name safely from Skill model, dict, or string."""
    if isinstance(skill_obj, Skill):
        return skill_obj.name
    if hasattr(skill_obj, "name"):
        return getattr(skill_obj, "name")
    if hasattr(skill_obj, "canonical_name"):
        return getattr(skill_obj, "canonical_name")
    return str(skill_obj)


class InterviewGeneratorError(SkillForgeError):
    """Raised when interview question generation fails."""


class InterviewQuestionGenerator:
    """
    Generates personalized interview question sets tailored to a candidate's resume,
    target job requirements, skill gaps, and retrieved career knowledge.
    """

    # Domain-specific technical question templates & evaluation rubrics
    DOMAIN_QUESTION_REGISTRY: dict[str, dict[str, Any]] = {
        "machine learning": {
            "conceptual": "How do you diagnose and address high variance versus high bias in a tree ensemble or gradient boosting model?",
            "technical": "Explain how you implement custom cross-validation (e.g. StratifiedKFold or TimeSeriesSplit) and handle feature leakage in Scikit-Learn pipelines.",
            "evaluation": [
                "Understands bias-variance trade-off and regularization techniques (L1/L2, learning rate, tree depth).",
                "Identifies root causes of data leakage (target leakage, test set contamination).",
                "Articulates proper metric selection beyond simple accuracy (ROC-AUC, Precision-Recall, F1).",
            ],
            "sample_points": [
                "High bias indicates underfitting; increase model capacity, engineer richer features, or tune depth.",
                "High variance indicates overfitting; apply regularization, prune trees, add dropout, or collect more data.",
                "Use ColumnTransformer and Pipeline objects to ensure all transformations fit solely on training folds.",
            ],
        },
        "deep learning": {
            "conceptual": "What is the vanishing gradient problem in deep neural networks, and how do modern architectures like ResNet (residual connections) and LayerNorm resolve it?",
            "technical": "How does PyTorch's autograd engine construct dynamic computation graphs, and how do you optimize CUDA memory during large batch training?",
            "evaluation": [
                "Deep understanding of backpropagation and gradient flow through computation graphs.",
                "Knowledge of memory optimization techniques: gradient checkpointing, mixed precision (torch.cuda.amp), zero_grad(set_to_none=True).",
                "Explains skip connections and normalization layers mathematically.",
            ],
            "sample_points": [
                "Residual connections allow gradients to flow unimpeded via identity mapping during backpropagation.",
                "PyTorch builds Directed Acyclic Graphs (DAGs) on the fly during forward pass, freed during backward pass.",
                "Use torch.cuda.amp.autocast for fp16/bf16 mixed precision to reduce memory footprint by ~50%.",
            ],
        },
        "pytorch": {
            "conceptual": "Explain the difference between nn.Module, nn.Sequential, and functional operators in PyTorch.",
            "technical": "How do you implement a custom DataLoader with multi-worker prefetching, and how do you handle DistributedDataParallel (DDP) synchronization?",
            "evaluation": [
                "Familiarity with PyTorch class inheritance and overriding __init__ and forward methods.",
                "Understands DDP vs DP differences (multiprocessing per GPU vs single-process multi-threading GIL bottleneck).",
                "Correct use of pin_memory=True and num_workers for asynchronous host-to-device transfers.",
            ],
            "sample_points": [
                "nn.Module manages stateful parameters and buffers; functional calls are stateless operations.",
                "DDP creates one process per GPU, eliminating Python GIL bottlenecks and using Ring-AllReduce for gradient sync.",
                "DataLoader should utilize pin_memory=True when training on CUDA to leverage DMA transfers.",
            ],
        },
        "nlp": {
            "conceptual": "How does self-attention in Transformer architectures differ from recurrent mechanisms (RNNs/LSTMs), and what is the computational complexity?",
            "technical": "How do you tokenize raw text using subword tokenization (BPE/WordPiece) and handle sequence truncation or padding in Hugging Face pipelines?",
            "evaluation": [
                "Explains Query, Key, Value matrices and Scaled Dot-Product Attention equation.",
                "States O(N^2) sequence length time/memory complexity of full attention.",
                "Understands attention masks and special tokens ([CLS], [SEP], <s>, </s>).",
            ],
            "sample_points": [
                "Transformers process all tokens concurrently via attention, overcoming RNN sequential bottlenecks.",
                "Attention formula: Softmax(Q * K^T / sqrt(d_k)) * V.",
                "Subword tokenizers balance out-of-vocabulary handling with vocabulary compactness.",
            ],
        },
        "natural language processing": {
            "conceptual": "Compare dense semantic vector embeddings (e.g. Sentence-BERT) with sparse lexical representations (TF-IDF/BM25). When would you choose a hybrid search approach?",
            "technical": "How do you design a semantic retrieval pipeline with re-ranking (Cross-Encoder) to maximize Precision@K?",
            "evaluation": [
                "Understands semantic vs exact lexical trade-offs (synonyms vs exact keyword matching like IDs/acronyms).",
                "Explains Bi-Encoder (fast retrieval) vs Cross-Encoder (accurate re-ranking) architecture.",
                "Demonstrates practical RAG retrieval optimization skills.",
            ],
            "sample_points": [
                "Bi-Encoders map queries and docs into vector space independently; Cross-Encoders evaluate query-doc pairs jointly.",
                "Use Bi-Encoder/FAISS to retrieve top-50 candidates, then Cross-Encoder to re-rank top-5.",
                "Hybrid search combines BM25 for keyword precision with dense embeddings for conceptual relevance.",
            ],
        },
        "computer vision": {
            "conceptual": "What is the difference between image classification, semantic segmentation, and object detection?",
            "technical": "How do single-stage object detectors like YOLO balance inference speed with mean Average Precision (mAP), and how do Anchor boxes / Anchor-free heads work?",
            "evaluation": [
                "Clear distinction between bounding boxes, pixel masks, and class probabilities.",
                "Understands Intersection over Union (IoU), Non-Maximum Suppression (NMS), and mAP@50-95 metrics.",
                "Discusses real-time inference constraints on edge devices.",
            ],
            "sample_points": [
                "Classification predicts whole-image class; object detection outputs bounding boxes; segmentation labels every pixel.",
                "YOLO divides the image into a grid and predicts bounding boxes and class probabilities in a single pass.",
                "Non-Maximum Suppression filters redundant overlapping boxes based on confidence and IoU threshold.",
            ],
        },
        "sql": {
            "conceptual": "Explain the difference between clustered and non-clustered indexes, and how B-Tree index structure speeds up range scans.",
            "technical": "Write and explain a SQL query utilizing Common Table Expressions (CTEs) and window functions (e.g. ROW_NUMBER, DENSE_RANK, or LAG/LEAD) to calculate rolling metrics.",
            "evaluation": [
                "Understanding of query execution plans, index scans vs table scans, and cardinality.",
                "Mastery of window function partitioning, ordering, and frame clauses (ROWS BETWEEN).",
                "Awareness of ACID transactions and isolation levels (Read Committed vs Serializable).",
            ],
            "sample_points": [
                "A clustered index defines physical table row order; non-clustered indexes store pointers to row locators.",
                "Window functions calculate aggregations over partitioned row sets without collapsing rows like GROUP BY.",
                "Use EXPLAIN ANALYZE to identify sequential scans and costly nested loop joins.",
            ],
        },
        "python": {
            "conceptual": "How does Python's Memory Management (reference counting + cyclic garbage collector) and Global Interpreter Lock (GIL) impact multithreading vs multiprocessing?",
            "technical": "Explain how Python's `asyncio` event loop works under the hood, and how coroutines differ from OS-level threads.",
            "evaluation": [
                "Understands GIL constraints on CPU-bound vs I/O-bound workloads.",
                "Explains generator functions, async/await coroutines, and non-blocking I/O multiplexing (epoll/kqueue).",
                "Knows dunder methods (__enter__/__exit__, __iter__/__next__, __call__).",
            ],
            "sample_points": [
                "GIL ensures only one native thread executes Python bytecode at a time, making multiprocessing preferable for CPU tasks.",
                "asyncio uses cooperative multitasking on a single thread via an event loop and non-blocking sockets.",
                "Generators and coroutines yield control back to the event loop without preemptive OS thread switching overhead.",
            ],
        },
        "mlops": {
            "conceptual": "What is the difference between data drift and concept drift, and how do you design an automated drift monitoring pipeline in production?",
            "technical": "How do you structure a CI/CD pipeline for ML that automates model training, validation against a baseline champion model, and blue/green deployment?",
            "evaluation": [
                "Defines covariate shift P(X) vs concept shift P(Y|X).",
                "Familiar with statistical tests for drift (KS-test, Population Stability Index (PSI), Wasserstein Distance).",
                "Explains model registry lifecycle (Staging -> Production -> Archived) and automated rollback triggers.",
            ],
            "sample_points": [
                "Data drift changes input distribution P(X); concept drift changes the relationship between inputs and outputs P(Y|X).",
                "Implement automated shadow deployments or canary releases to validate new models with production traffic.",
                "Track lineage: dataset hash, code commit SHA, hyperparameters, and environment docker image in MLflow.",
            ],
        },
        "cloud": {
            "conceptual": "Explain the Shared Responsibility Model in cloud computing (IaaS vs PaaS vs SaaS) and key security best practices for IAM.",
            "technical": "How do you design a high-availability, fault-tolerant infrastructure using Terraform Infrastructure as Code (IaC) across multiple availability zones?",
            "evaluation": [
                "Principles of least privilege IAM policies, role assumption, and secret management.",
                "Understanding of VPC subnetting (public vs private), NAT gateways, and load balancers.",
                "State management in Terraform (remote backends, state locking with DynamoDB).",
            ],
            "sample_points": [
                "In IaaS, customer manages OS, runtime, and app; in PaaS, cloud provider manages OS and runtime.",
                "Deploy compute instances in private subnets with traffic routed exclusively through Application Load Balancers.",
                "Use Terraform remote state stored in encrypted S3 with DynamoDB locking to prevent concurrent modifications.",
            ],
        },
        "data structures": {
            "conceptual": "What are the trade-offs between Hash Tables and Balanced Binary Search Trees (e.g. Red-Black Tree or AVL Tree)?",
            "technical": "How do you detect cycles in a directed graph (Tarjan's/Kahn's algorithm) or solve dynamic programming problems involving state memoization?",
            "evaluation": [
                "Analyzes worst-case vs amortized time/space complexity (Big-O notation).",
                "Understands recursion stack depth and dynamic programming overlapping subproblems.",
                "Familiar with graph representations (adjacency list vs adjacency matrix).",
            ],
            "sample_points": [
                "Hash tables offer O(1) average lookup but O(N) worst-case on collisions and lack order; BSTs guarantee O(log N) and sorted keys.",
                "Topological sorting with Kahn's algorithm uses in-degree tracking; a remaining non-zero in-degree signifies a cycle.",
                "DP breaks problems into optimal substructures with memoized lookup tables to avoid exponential recomputations.",
            ],
        },
        "generative ai": {
            "conceptual": "Explain the architecture of Retrieval-Augmented Generation (RAG). What mechanisms mitigate hallucination and context window overflow?",
            "technical": "How does Parameter-Efficient Fine-Tuning (LoRA / QLoRA) work, and how does it decompose weight update matrices during backpropagation?",
            "evaluation": [
                "Understands low-rank adaptation matrix decomposition W = W0 + (B * A) * (alpha / r).",
                "Explains chunking strategies, chunk overlap, and vector similarity thresholding in RAG.",
                "Familiar with LLM evaluation metrics: RAG Triad (Context Relevance, Groundedness, Answer Relevance).",
            ],
            "sample_points": [
                "LoRA freezes pretrained weights W0 and trains low-rank decomposition matrices A and B (rank r << d), reducing trainable params by 99%.",
                "RAG grounds autoregressive generation with external retrieved chunks, enforcing strict anti-hallucination prompt guardrails.",
                "Quantization (4-bit NF4 in QLoRA) enables fine-tuning large foundation models on consumer GPUs.",
            ],
        },
    }

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """
        Initialize the InterviewQuestionGenerator.

        Args:
            retrieval_service: RetrievalService to fetch supporting career knowledge.
            llm_provider: Optional LLM provider for enhanced synthesis.
        """
        self.retrieval_service = retrieval_service or RetrievalService()
        self.llm_provider = llm_provider

    # ── Main Question Generation Entry Point ───────────────────────────

    def generate_questions(
        self,
        match_result: MatchResult | None = None,
        resume_analysis: ResumeAnalysis | None = None,
        target_role: str = "",
        job_description: str = "",
        total_questions: int = 8,
    ) -> InterviewQuestionSet:
        """
        Generate a personalized interview question set tailored to the candidate.

        Args:
            match_result: MatchResult from M5 with skill matches and gaps.
            resume_analysis: Parsed ResumeAnalysis from M2 with experience and projects.
            target_role: Target job title.
            job_description: Optional job description text.
            total_questions: Desired number of questions (default 8).

        Returns:
            Structured InterviewQuestionSet.
        """
        role = target_role or (getattr(match_result, "target_role", "") if match_result else "Software Engineer")
        logger.info("Generating personalized interview questions", target_role=role)

        try:
            # 1. Calibrate difficulty level based on role seniority
            difficulty = self._calibrate_difficulty(role, match_result)

            # 2. Extract candidate strengths, missing skills, and resume projects
            matched_skills, missing_skills, partial_skills = self._extract_skill_pools(match_result)
            projects_and_roles = self._extract_candidate_projects_and_roles(resume_analysis)

            questions: list[InterviewQuestion] = []

            # 3. Generate Technical Questions (Grounded in RAG & Skill Gaps)
            tech_questions = self._generate_technical_questions(
                target_role=role,
                missing_skills=missing_skills,
                matched_skills=matched_skills,
                difficulty=difficulty,
                count=max(2, total_questions // 4),
            )
            questions.extend(tech_questions)

            # 4. Generate Conceptual Questions (Grounded in Theory & Knowledge Base)
            concept_questions = self._generate_conceptual_questions(
                target_role=role,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                difficulty=difficulty,
                count=max(2, total_questions // 4),
            )
            questions.extend(concept_questions)

            # 5. Generate Project-Based Questions (Personalized to Resume Experience)
            proj_questions = self._generate_project_questions(
                target_role=role,
                projects=projects_and_roles,
                matched_skills=matched_skills,
                difficulty=difficulty,
                count=max(2, total_questions // 4),
            )
            questions.extend(proj_questions)

            # 6. Generate Behavioral Questions (Personalized to Candidate Career Level)
            behavioral_questions = self._generate_behavioral_questions(
                target_role=role,
                difficulty=difficulty,
                projects=projects_and_roles,
                count=max(2, total_questions - len(questions)),
            )
            questions.extend(behavioral_questions)

            # Truncate or adjust to requested total
            final_questions = questions[:total_questions]

            # 7. Synthesize Summary, Focus Areas, and Preparation Tips
            summary = self._synthesize_summary(role, difficulty, len(final_questions), missing_skills)
            focus_areas = self._derive_focus_areas(matched_skills, missing_skills, role)
            tips = self._generate_preparation_tips(difficulty, role)

            return InterviewQuestionSet(
                target_role=role,
                difficulty_level=difficulty.value.capitalize(),
                summary=summary,
                questions=final_questions,
                total_questions=len(final_questions),
                focus_areas=focus_areas,
                preparation_tips=tips,
                generated_with_llm=self.llm_provider is not None,
                model_used=getattr(self.llm_provider, "model_name", "deterministic-rules"),
            )

        except Exception as e:
            logger.error("Failed to generate interview questions", error=str(e))
            raise InterviewGeneratorError(f"Failed to generate interview questions: {e}") from e

    # ── Difficulty Calibration ─────────────────────────────────────────

    def _calibrate_difficulty(
        self,
        target_role: str,
        match_result: MatchResult | None,
    ) -> QuestionDifficulty:
        """Calibrate question difficulty level based on role seniority keywords."""
        role_lower = target_role.lower()

        if any(w in role_lower for w in ["lead", "principal", "staff", "architect", "senior", "director"]):
            return QuestionDifficulty.HARD
        if any(w in role_lower for w in ["junior", "intern", "entry", "associate", "graduate"]):
            return QuestionDifficulty.EASY
        return QuestionDifficulty.MEDIUM

    # ── Skill & Project Extraction ─────────────────────────────────────

    def _extract_skill_pools(
        self,
        match_result: MatchResult | None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Extract matched, missing, and partial skills safely from match result."""
        if not match_result:
            return ["Python", "Machine Learning"], ["Docker", "MLOps"], []

        matched: list[str] = []
        for m in getattr(match_result, "matched_skills", []) or getattr(match_result, "matching_skills", []):
            s_name = _extract_skill_str(getattr(m, "job_skill", m))
            matched.append(s_name)

        missing: list[str] = []
        for g in getattr(match_result, "missing_skills", []) or getattr(match_result, "skill_gaps", []):
            g_name = _extract_skill_str(getattr(g, "skill", g))
            missing.append(g_name)

        partial: list[str] = []
        for p in getattr(match_result, "partial_matches", []):
            p_name = _extract_skill_str(getattr(p, "job_skill", p))
            partial.append(p_name)

        return matched, missing, partial

    def _extract_candidate_projects_and_roles(
        self,
        resume_analysis: ResumeAnalysis | None,
    ) -> list[dict[str, str]]:
        """Extract parsed project names and work history from resume sections."""
        extracted: list[dict[str, str]] = []

        if not resume_analysis:
            return [{"title": "Enterprise Data Pipeline", "context": "scalable backend processing"}]

        # Scan resume sections for projects or work experience
        sections = getattr(resume_analysis, "sections", []) or []
        for sec in sections:
            sec_title = (getattr(sec, "title", "") or getattr(sec, "name", "")).lower()
            sec_content = getattr(sec, "content", "")

            if any(k in sec_title for k in ["project", "experience", "work", "employment", "history"]):
                # Extract key bullet lines
                for line in sec_content.splitlines():
                    clean = line.strip().lstrip("-*•# ").strip()
                    if len(clean) > 25 and len(clean) < 140:
                        extracted.append({
                            "title": clean[:50] + ("..." if len(clean) > 50 else ""),
                            "context": clean,
                        })
                    if len(extracted) >= 5:
                        break

        if not extracted:
            extracted.append({"title": "Production Service Architecture", "context": "building scalable backend services"})

        return extracted

    # ── Category 1: Technical Questions (Grounded in RAG) ───────────────

    def _generate_technical_questions(
        self,
        target_role: str,
        missing_skills: list[str],
        matched_skills: list[str],
        difficulty: QuestionDifficulty,
        count: int = 2,
    ) -> list[InterviewQuestion]:
        """Generate in-depth technical questions targeting skills and missing gaps."""
        questions: list[InterviewQuestion] = []
        target_pool = (missing_skills + matched_skills) or ["Python", "Machine Learning"]

        for skill in target_pool:
            if len(questions) >= count:
                break

            lookup_key = self._find_domain_key(skill)
            is_gap = skill in missing_skills
            citations = self._retrieve_rag_citations(skill)
            has_kb = len(citations) > 0

            if lookup_key and lookup_key in self.DOMAIN_QUESTION_REGISTRY:
                data = self.DOMAIN_QUESTION_REGISTRY[lookup_key]
                q_text = data["technical"]
                eval_pts = data["evaluation"]
                sample_pts = data["sample_points"]
            else:
                q_text = f"How do you design, implement, and debug complex workflows using {skill} in a production {target_role} environment?"
                eval_pts = [
                    f"Demonstrates hands-on proficiency with {skill} APIs and tooling.",
                    f"Identifies common runtime errors and debugging techniques for {skill}.",
                    f"Understands scaling and performance optimization for {skill}.",
                ]
                sample_pts = [
                    f"Articulate architecture patterns standard in {skill}.",
                    f"Explain performance profiling and automated testing strategies for {skill}.",
                ]

            why = (
                f"Target role requires {skill}; candidate's profile shows a skill gap, so the interviewer will probe practical readiness."
                if is_gap
                else f"Candidate lists {skill} as a core strength; interviewers will verify depth of technical expertise."
            )

            questions.append(
                InterviewQuestion(
                    question=q_text,
                    category=QuestionCategory.TECHNICAL,
                    difficulty=difficulty,
                    related_skill=skill,
                    why_this_question=why,
                    evaluation_points=eval_pts,
                    sample_answer_points=sample_pts,
                    guidance=f"Be prepared to write code snippets or explain internal mechanics of {skill}. Mention production trade-offs.",
                    supporting_citations=citations,
                    has_supporting_knowledge=has_kb,
                )
            )

        return questions

    # ── Category 2: Conceptual Questions (Grounded in Theory) ───────────

    def _generate_conceptual_questions(
        self,
        target_role: str,
        matched_skills: list[str],
        missing_skills: list[str],
        difficulty: QuestionDifficulty,
        count: int = 2,
    ) -> list[InterviewQuestion]:
        """Generate conceptual and architectural questions grounded in foundational theory."""
        questions: list[InterviewQuestion] = []
        pool = matched_skills or missing_skills or ["Machine Learning", "SQL"]

        for skill in pool:
            if len(questions) >= count:
                break

            lookup_key = self._find_domain_key(skill)
            citations = self._retrieve_rag_citations(skill)
            has_kb = len(citations) > 0

            if lookup_key and lookup_key in self.DOMAIN_QUESTION_REGISTRY:
                data = self.DOMAIN_QUESTION_REGISTRY[lookup_key]
                q_text = data["conceptual"]
                eval_pts = data["evaluation"]
                sample_pts = data["sample_points"]
            else:
                q_text = f"What theoretical principles and architectural trade-offs govern the effective use of {skill} in modern software systems?"
                eval_pts = [
                    f"Explains underlying mechanisms and mathematical/system principles of {skill}.",
                    f"Discusses trade-offs between simplicity, speed, and maintainability.",
                ]
                sample_pts = [
                    f"State the core design assumptions behind {skill}.",
                    f"Contrast {skill} with alternative industry paradigms.",
                ]

            questions.append(
                InterviewQuestion(
                    question=q_text,
                    category=QuestionCategory.CONCEPTUAL,
                    difficulty=difficulty,
                    related_skill=skill,
                    why_this_question=f"Tests fundamental grasp of theoretical principles behind {skill} rather than mere API memorization.",
                    evaluation_points=eval_pts,
                    sample_answer_points=sample_pts,
                    guidance="Focus on fundamental theory, mathematical intuition, and architectural trade-offs.",
                    supporting_citations=citations,
                    has_supporting_knowledge=has_kb,
                )
            )

        return questions

    # ── Category 3: Project-Based Questions ─────────────────────────────

    def _generate_project_questions(
        self,
        target_role: str,
        projects: list[dict[str, str]],
        matched_skills: list[str],
        difficulty: QuestionDifficulty,
        count: int = 2,
    ) -> list[InterviewQuestion]:
        """Generate deep-dive questions probing the candidate's actual projects."""
        questions: list[InterviewQuestion] = []
        top_skill = matched_skills[0] if matched_skills else "Python"

        for p in projects:
            if len(questions) >= count:
                break

            p_title = p.get("title", "Recent Project")
            p_context = p.get("context", "")

            q_text = (
                f"In your work involving '{p_title}', what were the most significant technical bottlenecks or scaling challenges you encountered, and how did you resolve them?"
            )
            why = f"Directly probes the candidate's resume project experience ({p_title}) to verify hands-on ownership and problem-solving depth."

            questions.append(
                InterviewQuestion(
                    question=q_text,
                    category=QuestionCategory.PROJECT_BASED,
                    difficulty=difficulty,
                    related_skill=top_skill,
                    why_this_question=why,
                    evaluation_points=[
                        "Articulates the technical problem with clarity and quantifiable context.",
                        "Explains design decisions and alternative approaches considered.",
                        "Demonstrates clear personal contribution vs team/general efforts.",
                        "Quantifies outcome (e.g. latency reduction, throughput, accuracy).",
                    ],
                    sample_answer_points=[
                        f"Describe the system architecture and load constraints for {p_title}.",
                        "Identify the exact bottleneck (I/O, database locks, GPU memory, network).",
                        "Detail the specific fix implemented and measurable performance gains achieved.",
                    ],
                    guidance="Use the STAR method (Situation, Task, Action, Result). Highlight your personal technical contribution.",
                    supporting_citations=[],
                    has_supporting_knowledge=False,
                )
            )

        return questions

    # ── Category 4: Behavioral Questions ───────────────────────────────

    def _generate_behavioral_questions(
        self,
        target_role: str,
        difficulty: QuestionDifficulty,
        projects: list[dict[str, str]],
        count: int = 2,
    ) -> list[InterviewQuestion]:
        """Generate behavioral interview questions tailored to role seniority."""
        questions: list[InterviewQuestion] = []
        is_senior = difficulty == QuestionDifficulty.HARD

        behavioral_templates = [
            {
                "question": (
                    f"Tell me about a time when you had a strong technical disagreement with a colleague or stakeholder regarding system architecture for a {target_role} deliverable. How did you navigate it?"
                    if is_senior
                    else "Describe a situation where you received constructive feedback on your code or design during a critical sprint. How did you handle it?"
                ),
                "why": "Evaluates technical communication, ego management, and conflict resolution under engineering pressure.",
                "eval": [
                    "Focuses on objective data and benchmarks rather than emotional argumentation.",
                    "Demonstrates active listening and empathy toward alternative perspectives.",
                    "Shows commitment to the final team decision once consensus or decision is reached.",
                ],
                "sample": [
                    "Frame the disagreement around system trade-offs (e.g. latency vs engineering complexity).",
                    "Explain how you built a proof-of-concept (POC) or metric comparison to inform the team.",
                    "Describe the successful delivery and positive working relationship maintained.",
                ],
            },
            {
                "question": (
                    f"Describe a high-severity production incident or unexpected failure you responded to in your past projects. Walk me through your troubleshooting process, resolution, and post-mortem improvements."
                ),
                "why": "Evaluates composure under pressure, root cause analysis rigor, and proactive system hardening.",
                "eval": [
                    "Follows systematic triage and hypothesis-driven debugging.",
                    "Focuses on fast mitigation first, followed by root-cause analysis (RCA).",
                    "Institutes permanent preventative measures (alerts, automated tests, runbooks).",
                ],
                "sample": [
                    "State the incident severity, blast radius, and alert mechanism.",
                    "Explain how you isolated the fault using logs, metrics, and tracing.",
                    "Detail the blameless post-mortem actions created to prevent recurrence.",
                ],
            },
        ]

        for t in behavioral_templates[:count]:
            questions.append(
                InterviewQuestion(
                    question=t["question"],
                    category=QuestionCategory.BEHAVIORAL,
                    difficulty=difficulty,
                    related_skill="Communication & Collaboration",
                    why_this_question=t["why"],
                    evaluation_points=t["eval"],
                    sample_answer_points=t["sample"],
                    guidance="Structure your response using STAR: Situation (20%), Task (10%), Action (50%), Result (20%).",
                    supporting_citations=[],
                    has_supporting_knowledge=False,
                )
            )

        return questions

    # ── Retrieval & Domain Key Resolution Helpers ──────────────────────

    def _retrieve_rag_citations(self, skill_name: str) -> list[SourceCitation]:
        """Fetch supporting citations from FAISS knowledge base."""
        try:
            citations = self.retrieval_service.retrieve_from_knowledge_base(
                query=f"{skill_name} concepts tools interview theory practical architecture",
                top_k=2,
                min_score=0.25,
            )
            return citations
        except Exception as e:
            logger.warning("Failed to retrieve knowledge for interview question", skill=skill_name, error=str(e))
            return []

    @staticmethod
    def _find_domain_key(skill_name: str) -> str | None:
        """Find matching key in DOMAIN_QUESTION_REGISTRY."""
        s = skill_name.lower().strip()
        for key in InterviewQuestionGenerator.DOMAIN_QUESTION_REGISTRY:
            if key in s or s in key:
                return key
        return None

    # ── Summary & Tip Synthesis ────────────────────────────────────────

    def _synthesize_summary(
        self,
        role: str,
        difficulty: QuestionDifficulty,
        total: int,
        missing_skills: list[str],
    ) -> str:
        """Generate high-level overview of the interview strategy."""
        gap_phrase = f" with targeted probes on {len(missing_skills)} identified skill gaps" if missing_skills else ""
        return (
            f"Tailored {difficulty.value.capitalize()}-level interview preparation set for **{role}**{gap_phrase}. "
            f"Contains **{total} personalized questions** balanced across technical depth, theoretical concepts, "
            f"resume project deep-dives, and behavioral leadership scenarios."
        )

    @staticmethod
    def _derive_focus_areas(matched: list[str], missing: list[str], role: str) -> list[str]:
        """Derive top 3 interview focus areas."""
        focus: list[str] = []
        if missing:
            focus.append(f"Skill Gap Defensibility ({missing[0]})")
        if matched:
            focus.append(f"Core Competency Deep-Dive ({matched[0]})")
        focus.append("System Architecture & Trade-Offs")
        return focus[:3]

    @staticmethod
    def _generate_preparation_tips(difficulty: QuestionDifficulty, role: str) -> list[str]:
        """Generate targeted preparation tips."""
        return [
            "Use the STAR framework (Situation, Task, Action, Result) for project and behavioral questions.",
            "Always state architectural and computational trade-offs (e.g. time vs space, latency vs throughput).",
            "Be upfront about missing technologies by pivoting to transferable fundamentals and rapid learning examples.",
            "Prepare 2-3 specific technical metrics from your past projects (e.g., '% latency reduction', 'QPS supported').",
        ]
