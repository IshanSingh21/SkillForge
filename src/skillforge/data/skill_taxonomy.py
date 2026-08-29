"""
SkillForge AI — Curated Skill Taxonomy.

A structured database of ~300 technology and professional skills with
canonical names, aliases, and categories. This taxonomy is the backbone
of pattern-based skill extraction — it provides reliable, deterministic
matching before any ML-based techniques are applied.

The taxonomy is designed to be:
    - Comprehensive: covers major languages, frameworks, tools, clouds, DBs,
      data/ML, DevOps, and soft skills.
    - Alias-aware: "ReactJS", "React.js", "React" all map to "React".
    - Boundary-safe: short names like "C", "R", "Go" use word-boundary
      matching to avoid false positives.
    - Extensible: add new skills by appending to SKILL_DEFINITIONS.

Usage:
    from src.skillforge.data.skill_taxonomy import SkillTaxonomy

    taxonomy = SkillTaxonomy()
    definition = taxonomy.lookup("react.js")  # → SkillDefinition(name="React", ...)
    all_dbs = taxonomy.get_by_category("database")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from src.skillforge.models.resume import SkillCategory


@dataclass(frozen=True)
class SkillDefinition:
    """A single skill entry in the taxonomy."""

    name: str                          # Canonical name (e.g., "Python")
    category: SkillCategory            # Broad category
    aliases: tuple[str, ...] = ()      # Alternative spellings/names
    requires_word_boundary: bool = False  # Use \\b matching (for "C", "R", "Go", etc.)


# ── Taxonomy Data ──────────────────────────────────────────────────────
# Format: (canonical_name, category, aliases_tuple, requires_boundary)
# Aliases are LOWERCASE — lookup normalizes to lowercase before matching.

_RAW_SKILLS: list[tuple[str, SkillCategory, tuple[str, ...], bool]] = [
    # ── Programming Languages ──────────────────────────────────────
    ("Python", SkillCategory.LANGUAGE, ("python3", "python 3", "py"), False),
    ("JavaScript", SkillCategory.LANGUAGE, ("js", "ecmascript", "es6", "es2015"), False),
    ("TypeScript", SkillCategory.LANGUAGE, ("ts",), False),
    ("Java", SkillCategory.LANGUAGE, (), False),
    ("C++", SkillCategory.LANGUAGE, ("cpp", "c plus plus", "cplusplus"), False),
    ("C#", SkillCategory.LANGUAGE, ("csharp", "c sharp", "c-sharp"), False),
    ("C", SkillCategory.LANGUAGE, (), True),  # Requires boundary
    ("Go", SkillCategory.LANGUAGE, ("golang",), True),
    ("Rust", SkillCategory.LANGUAGE, ("rust-lang", "rustlang"), False),
    ("Ruby", SkillCategory.LANGUAGE, (), False),
    ("PHP", SkillCategory.LANGUAGE, ("php7", "php8"), False),
    ("Swift", SkillCategory.LANGUAGE, (), False),
    ("Kotlin", SkillCategory.LANGUAGE, (), False),
    ("Scala", SkillCategory.LANGUAGE, (), False),
    ("R", SkillCategory.LANGUAGE, ("r language", "r programming", "rlang"), True),
    ("SQL", SkillCategory.LANGUAGE, ("structured query language",), False),
    ("Bash", SkillCategory.LANGUAGE, ("shell scripting", "shell script", "bash scripting", "zsh"), False),
    ("Perl", SkillCategory.LANGUAGE, (), False),
    ("MATLAB", SkillCategory.LANGUAGE, (), False),
    ("Dart", SkillCategory.LANGUAGE, (), False),
    ("Lua", SkillCategory.LANGUAGE, (), True),
    ("Haskell", SkillCategory.LANGUAGE, (), False),
    ("Elixir", SkillCategory.LANGUAGE, (), False),
    ("Clojure", SkillCategory.LANGUAGE, (), False),
    ("Groovy", SkillCategory.LANGUAGE, (), False),
    ("Julia", SkillCategory.LANGUAGE, (), False),
    ("Objective-C", SkillCategory.LANGUAGE, ("objectivec", "objective c", "obj-c"), False),
    ("Assembly", SkillCategory.LANGUAGE, ("asm", "assembly language"), False),
    ("Solidity", SkillCategory.LANGUAGE, (), False),
    ("HTML", SkillCategory.LANGUAGE, ("html5",), False),
    ("CSS", SkillCategory.LANGUAGE, ("css3", "cascading style sheets"), False),
    ("SASS", SkillCategory.LANGUAGE, ("scss",), False),

    # ── Frontend Frameworks ────────────────────────────────────────
    ("React", SkillCategory.FRAMEWORK, ("reactjs", "react.js", "react js", "react native"), False),
    ("Angular", SkillCategory.FRAMEWORK, ("angularjs", "angular.js", "angular js"), False),
    ("Vue.js", SkillCategory.FRAMEWORK, ("vue", "vuejs", "vue js"), False),
    ("Svelte", SkillCategory.FRAMEWORK, ("sveltejs", "sveltekit"), False),
    ("Next.js", SkillCategory.FRAMEWORK, ("nextjs", "next js", "next"), False),
    ("Nuxt.js", SkillCategory.FRAMEWORK, ("nuxtjs", "nuxt"), False),
    ("jQuery", SkillCategory.FRAMEWORK, ("jquery",), False),
    ("Bootstrap", SkillCategory.FRAMEWORK, ("bootstrap 5", "bootstrap5"), False),
    ("Tailwind CSS", SkillCategory.FRAMEWORK, ("tailwind", "tailwindcss"), False),
    ("Material UI", SkillCategory.FRAMEWORK, ("mui", "material-ui", "material design"), False),

    # ── Backend Frameworks ─────────────────────────────────────────
    ("Django", SkillCategory.FRAMEWORK, ("django rest framework", "drf"), False),
    ("Flask", SkillCategory.FRAMEWORK, (), False),
    ("FastAPI", SkillCategory.FRAMEWORK, ("fast api", "fast-api"), False),
    ("Express.js", SkillCategory.FRAMEWORK, ("express", "expressjs"), False),
    ("Spring Boot", SkillCategory.FRAMEWORK, ("spring", "spring framework", "spring mvc"), False),
    ("Ruby on Rails", SkillCategory.FRAMEWORK, ("rails", "ror"), False),
    ("Laravel", SkillCategory.FRAMEWORK, (), False),
    ("NestJS", SkillCategory.FRAMEWORK, ("nest.js", "nest js"), False),
    ("ASP.NET", SkillCategory.FRAMEWORK, ("asp.net core", "aspnet", "dotnet", ".net", ".net core"), False),
    ("Gin", SkillCategory.FRAMEWORK, ("gin-gonic",), True),
    ("Phoenix", SkillCategory.FRAMEWORK, ("phoenix framework",), False),
    ("Tornado", SkillCategory.FRAMEWORK, (), False),
    ("Celery", SkillCategory.FRAMEWORK, (), False),
    ("Streamlit", SkillCategory.FRAMEWORK, (), False),
    ("Gradio", SkillCategory.FRAMEWORK, (), False),

    # ── Databases ──────────────────────────────────────────────────
    ("PostgreSQL", SkillCategory.TOOL, ("postgres", "psql", "pg"), False),
    ("MySQL", SkillCategory.TOOL, ("my sql",), False),
    ("MongoDB", SkillCategory.TOOL, ("mongo", "mongo db"), False),
    ("Redis", SkillCategory.TOOL, (), False),
    ("Elasticsearch", SkillCategory.TOOL, ("elastic search", "elastic", "es", "elk"), False),
    ("DynamoDB", SkillCategory.TOOL, ("dynamo db", "dynamodb", "amazon dynamodb"), False),
    ("Cassandra", SkillCategory.TOOL, ("apache cassandra",), False),
    ("SQLite", SkillCategory.TOOL, ("sqlite3",), False),
    ("MariaDB", SkillCategory.TOOL, ("maria db",), False),
    ("Oracle", SkillCategory.TOOL, ("oracle db", "oracle database"), False),
    ("SQL Server", SkillCategory.TOOL, ("mssql", "microsoft sql server", "ms sql"), False),
    ("Neo4j", SkillCategory.TOOL, (), False),
    ("InfluxDB", SkillCategory.TOOL, ("influx db",), False),
    ("CouchDB", SkillCategory.TOOL, ("couch db",), False),
    ("Firebase", SkillCategory.TOOL, ("firebase realtime database", "firestore"), False),
    ("Supabase", SkillCategory.TOOL, (), False),

    # ── Cloud Platforms & Services ─────────────────────────────────
    ("AWS", SkillCategory.TOOL, ("amazon web services", "amazon aws"), False),
    ("Google Cloud", SkillCategory.TOOL, ("gcp", "google cloud platform"), False),
    ("Azure", SkillCategory.TOOL, ("microsoft azure", "azure cloud"), False),
    ("Heroku", SkillCategory.TOOL, (), False),
    ("DigitalOcean", SkillCategory.TOOL, ("digital ocean",), False),
    ("Vercel", SkillCategory.TOOL, (), False),
    ("Netlify", SkillCategory.TOOL, (), False),
    ("Cloudflare", SkillCategory.TOOL, (), False),
    ("AWS Lambda", SkillCategory.TOOL, ("lambda", "serverless lambda"), False),
    ("AWS S3", SkillCategory.TOOL, ("s3", "amazon s3", "simple storage service"), True),
    ("AWS EC2", SkillCategory.TOOL, ("ec2",), False),
    ("AWS ECS", SkillCategory.TOOL, ("ecs", "elastic container service"), False),
    ("AWS EKS", SkillCategory.TOOL, ("eks",), False),
    ("AWS RDS", SkillCategory.TOOL, ("rds",), False),
    ("AWS SQS", SkillCategory.TOOL, ("sqs",), False),
    ("AWS SNS", SkillCategory.TOOL, ("sns",), False),
    ("AWS CloudFormation", SkillCategory.TOOL, ("cloudformation",), False),
    ("Google BigQuery", SkillCategory.TOOL, ("bigquery", "big query"), False),
    ("Google Kubernetes Engine", SkillCategory.TOOL, ("gke",), False),
    ("Azure DevOps", SkillCategory.TOOL, (), False),

    # ── DevOps & Infrastructure ────────────────────────────────────
    ("Docker", SkillCategory.TOOL, ("docker compose", "dockerfile", "docker-compose"), False),
    ("Kubernetes", SkillCategory.TOOL, ("k8s", "kube"), False),
    ("Terraform", SkillCategory.TOOL, ("terraform cloud", "hcl"), False),
    ("Ansible", SkillCategory.TOOL, (), False),
    ("Puppet", SkillCategory.TOOL, (), False),
    ("Chef", SkillCategory.TOOL, ("chef infra",), True),
    ("Jenkins", SkillCategory.TOOL, (), False),
    ("GitHub Actions", SkillCategory.TOOL, ("github action", "gh actions"), False),
    ("GitLab CI", SkillCategory.TOOL, ("gitlab ci/cd", "gitlab-ci", "gitlab ci cd"), False),
    ("CircleCI", SkillCategory.TOOL, ("circle ci",), False),
    ("Travis CI", SkillCategory.TOOL, ("travisci", "travis-ci"), False),
    ("ArgoCD", SkillCategory.TOOL, ("argo cd", "argo"), False),
    ("Prometheus", SkillCategory.TOOL, (), False),
    ("Grafana", SkillCategory.TOOL, (), False),
    ("Datadog", SkillCategory.TOOL, ("data dog",), False),
    ("New Relic", SkillCategory.TOOL, ("newrelic",), False),
    ("Nginx", SkillCategory.TOOL, ("nginx",), False),
    ("Apache", SkillCategory.TOOL, ("apache httpd", "apache http server"), False),
    ("Linux", SkillCategory.TOOL, ("linux administration", "ubuntu", "centos", "debian", "rhel"), False),
    ("Unix", SkillCategory.TOOL, (), False),
    ("Vagrant", SkillCategory.TOOL, (), False),
    ("Helm", SkillCategory.TOOL, ("helm charts",), True),
    ("Istio", SkillCategory.TOOL, (), False),

    # ── Version Control & Collaboration ────────────────────────────
    ("Git", SkillCategory.TOOL, ("git version control",), True),
    ("GitHub", SkillCategory.TOOL, (), False),
    ("GitLab", SkillCategory.TOOL, (), False),
    ("Bitbucket", SkillCategory.TOOL, ("bit bucket",), False),
    ("Jira", SkillCategory.TOOL, ("jira software",), False),
    ("Confluence", SkillCategory.TOOL, (), False),
    ("Slack", SkillCategory.TOOL, (), False),
    ("Notion", SkillCategory.TOOL, (), False),

    # ── Data & ML ──────────────────────────────────────────────────
    ("TensorFlow", SkillCategory.FRAMEWORK, ("tensorflow 2", "tf", "tensorflow.js"), False),
    ("PyTorch", SkillCategory.FRAMEWORK, ("pytorch lightning", "torch"), False),
    ("scikit-learn", SkillCategory.FRAMEWORK, ("sklearn", "scikit learn"), False),
    ("Keras", SkillCategory.FRAMEWORK, (), False),
    ("pandas", SkillCategory.FRAMEWORK, (), False),
    ("NumPy", SkillCategory.FRAMEWORK, ("numpy",), False),
    ("SciPy", SkillCategory.FRAMEWORK, ("scipy",), False),
    ("Matplotlib", SkillCategory.FRAMEWORK, (), False),
    ("Seaborn", SkillCategory.FRAMEWORK, (), False),
    ("Plotly", SkillCategory.FRAMEWORK, (), False),
    ("Hugging Face", SkillCategory.FRAMEWORK, ("huggingface", "hf", "transformers library"), False),
    ("LangChain", SkillCategory.FRAMEWORK, ("langchain",), False),
    ("OpenCV", SkillCategory.FRAMEWORK, ("cv2", "open cv"), False),
    ("NLTK", SkillCategory.FRAMEWORK, ("natural language toolkit",), False),
    ("spaCy", SkillCategory.FRAMEWORK, ("spacy",), False),
    ("XGBoost", SkillCategory.FRAMEWORK, ("xg boost",), False),
    ("LightGBM", SkillCategory.FRAMEWORK, ("light gbm",), False),
    ("MLflow", SkillCategory.FRAMEWORK, ("ml flow", "mlflow"), False),
    ("Apache Spark", SkillCategory.FRAMEWORK, ("spark", "pyspark"), False),
    ("Apache Kafka", SkillCategory.FRAMEWORK, ("kafka",), False),
    ("Apache Airflow", SkillCategory.FRAMEWORK, ("airflow",), False),
    ("Apache Flink", SkillCategory.FRAMEWORK, ("flink",), False),
    ("dbt", SkillCategory.TOOL, ("data build tool",), True),
    ("Tableau", SkillCategory.TOOL, (), False),
    ("Power BI", SkillCategory.TOOL, ("powerbi", "power-bi"), False),
    ("Looker", SkillCategory.TOOL, (), False),
    ("Jupyter", SkillCategory.TOOL, ("jupyter notebook", "jupyter lab", "jupyterlab"), False),
    ("FAISS", SkillCategory.TOOL, (), False),
    ("Pinecone", SkillCategory.TOOL, (), False),
    ("Weaviate", SkillCategory.TOOL, (), False),
    ("ChromaDB", SkillCategory.TOOL, ("chroma", "chroma db"), False),

    # ── Build Tools & Package Managers ─────────────────────────────
    ("Webpack", SkillCategory.TOOL, (), False),
    ("Vite", SkillCategory.TOOL, (), False),
    ("npm", SkillCategory.TOOL, (), True),
    ("yarn", SkillCategory.TOOL, (), True),
    ("pip", SkillCategory.TOOL, (), True),
    ("Maven", SkillCategory.TOOL, (), False),
    ("Gradle", SkillCategory.TOOL, (), False),
    ("CMake", SkillCategory.TOOL, ("cmake",), False),
    ("Poetry", SkillCategory.TOOL, (), False),

    # ── Testing ────────────────────────────────────────────────────
    ("pytest", SkillCategory.TOOL, ("py.test",), False),
    ("Jest", SkillCategory.TOOL, (), False),
    ("Mocha", SkillCategory.TOOL, (), False),
    ("Cypress", SkillCategory.TOOL, (), False),
    ("Selenium", SkillCategory.TOOL, ("selenium webdriver",), False),
    ("Playwright", SkillCategory.TOOL, (), False),
    ("JUnit", SkillCategory.TOOL, ("junit5", "junit 5"), False),
    ("unittest", SkillCategory.TOOL, ("unit test",), False),

    # ── APIs & Protocols ───────────────────────────────────────────
    ("REST", SkillCategory.TECHNICAL, ("restful", "rest api", "restful api", "rest apis"), False),
    ("GraphQL", SkillCategory.TECHNICAL, ("graph ql",), False),
    ("gRPC", SkillCategory.TECHNICAL, ("grpc",), False),
    ("WebSocket", SkillCategory.TECHNICAL, ("websockets", "web socket", "web sockets"), False),
    ("OAuth", SkillCategory.TECHNICAL, ("oauth2", "oauth 2.0", "oauth2.0"), False),
    ("JWT", SkillCategory.TECHNICAL, ("json web token", "json web tokens"), False),

    # ── Architectural Concepts ─────────────────────────────────────
    ("Microservices", SkillCategory.TECHNICAL, ("microservice", "micro services", "microservice architecture"), False),
    ("CI/CD", SkillCategory.TECHNICAL, ("ci cd", "cicd", "continuous integration", "continuous delivery", "continuous deployment"), False),
    ("Agile", SkillCategory.TECHNICAL, ("agile methodology", "agile development"), False),
    ("Scrum", SkillCategory.TECHNICAL, ("scrum master", "scrum methodology"), False),
    ("Kanban", SkillCategory.TECHNICAL, (), False),
    ("DevOps", SkillCategory.TECHNICAL, ("dev ops",), False),
    ("MLOps", SkillCategory.TECHNICAL, ("ml ops", "mlops"), False),
    ("DataOps", SkillCategory.TECHNICAL, ("data ops",), False),
    ("Infrastructure as Code", SkillCategory.TECHNICAL, ("iac", "infra as code"), False),
    ("Event-Driven Architecture", SkillCategory.TECHNICAL, ("event driven", "eda"), False),
    ("Domain-Driven Design", SkillCategory.TECHNICAL, ("ddd",), False),
    ("Test-Driven Development", SkillCategory.TECHNICAL, ("tdd",), False),
    ("Object-Oriented Programming", SkillCategory.TECHNICAL, ("oop", "object oriented"), False),
    ("Functional Programming", SkillCategory.TECHNICAL, ("fp", "functional paradigm"), False),
    ("Design Patterns", SkillCategory.TECHNICAL, ("software design patterns",), False),
    ("System Design", SkillCategory.TECHNICAL, ("systems design", "system architecture"), False),
    ("Data Structures", SkillCategory.TECHNICAL, ("data structures and algorithms", "dsa"), False),
    ("Algorithms", SkillCategory.TECHNICAL, ("algorithm design",), False),

    # ── Domain Knowledge ───────────────────────────────────────────
    ("Machine Learning", SkillCategory.DOMAIN, ("ml",), False),
    ("Deep Learning", SkillCategory.DOMAIN, ("dl", "deep neural networks"), False),
    ("Natural Language Processing", SkillCategory.DOMAIN, ("nlp",), False),
    ("Computer Vision", SkillCategory.DOMAIN, ("cv", "image recognition", "image processing"), False),
    ("Data Science", SkillCategory.DOMAIN, ("data analytics",), False),
    ("Data Engineering", SkillCategory.DOMAIN, ("data pipeline", "data pipelines", "etl"), False),
    ("Artificial Intelligence", SkillCategory.DOMAIN, ("ai",), False),
    ("Generative AI", SkillCategory.DOMAIN, ("genai", "gen ai", "generative artificial intelligence"), False),
    ("Large Language Models", SkillCategory.DOMAIN, ("llm", "llms"), False),
    ("Retrieval-Augmented Generation", SkillCategory.DOMAIN, ("rag",), False),
    ("Prompt Engineering", SkillCategory.DOMAIN, ("prompt design",), False),
    ("Reinforcement Learning", SkillCategory.DOMAIN, ("rl",), False),
    ("Recommendation Systems", SkillCategory.DOMAIN, ("recommender systems", "recommendation engine"), False),
    ("Time Series Analysis", SkillCategory.DOMAIN, ("time series", "time series forecasting"), False),
    ("A/B Testing", SkillCategory.DOMAIN, ("ab testing", "a-b testing", "split testing"), False),
    ("Cloud Computing", SkillCategory.DOMAIN, ("cloud architecture", "cloud infrastructure"), False),
    ("Cybersecurity", SkillCategory.DOMAIN, ("cyber security", "information security", "infosec"), False),
    ("Blockchain", SkillCategory.DOMAIN, ("web3", "web 3"), False),
    ("IoT", SkillCategory.DOMAIN, ("internet of things",), False),
    ("Embedded Systems", SkillCategory.DOMAIN, ("embedded programming",), False),

    # ── Certifications ─────────────────────────────────────────────
    ("AWS Solutions Architect", SkillCategory.CERTIFICATION, ("aws sa", "aws solutions architect associate"), False),
    ("AWS Developer Associate", SkillCategory.CERTIFICATION, ("aws developer",), False),
    ("Google Cloud Professional", SkillCategory.CERTIFICATION, ("gcp professional",), False),
    ("Azure Administrator", SkillCategory.CERTIFICATION, ("az-104",), False),
    ("Kubernetes Administrator", SkillCategory.CERTIFICATION, ("cka", "certified kubernetes administrator"), False),
    ("PMP", SkillCategory.CERTIFICATION, ("project management professional",), False),
    ("Scrum Master Certification", SkillCategory.CERTIFICATION, ("csm", "certified scrum master"), False),

    # ── Soft Skills ────────────────────────────────────────────────
    ("Leadership", SkillCategory.SOFT, ("team leadership", "tech lead", "team lead", "leading teams"), False),
    ("Communication", SkillCategory.SOFT, ("written communication", "verbal communication", "technical communication"), False),
    ("Teamwork", SkillCategory.SOFT, ("collaboration", "team player", "cross-functional", "cross functional"), False),
    ("Problem Solving", SkillCategory.SOFT, ("problem-solving", "analytical thinking", "critical thinking"), False),
    ("Project Management", SkillCategory.SOFT, ("project planning", "program management"), False),
    ("Mentoring", SkillCategory.SOFT, ("mentorship", "coaching", "training", "mentored"), False),
    ("Time Management", SkillCategory.SOFT, ("time-management", "prioritization"), False),
    ("Adaptability", SkillCategory.SOFT, ("flexibility", "adaptable"), False),
    ("Presentation Skills", SkillCategory.SOFT, ("public speaking", "presenting"), False),
    ("Technical Writing", SkillCategory.SOFT, ("documentation", "technical documentation"), False),
    ("Code Review", SkillCategory.SOFT, ("code reviews", "peer review"), False),
    ("Stakeholder Management", SkillCategory.SOFT, ("stakeholder communication",), False),
]


class SkillTaxonomy:
    """
    Searchable skill taxonomy with alias resolution.

    Builds an inverted index from all aliases and canonical names
    to SkillDefinition objects for O(1) lookup.
    """

    def __init__(self, extra_skills: list[SkillDefinition] | None = None) -> None:
        """
        Initialize the taxonomy from the built-in definitions.

        Args:
            extra_skills: Additional skills to include beyond the built-in set.
        """
        self._skills: list[SkillDefinition] = []
        self._lookup_index: dict[str, SkillDefinition] = {}  # lowercase name/alias → definition

        # Build from raw data
        for name, category, aliases, requires_boundary in _RAW_SKILLS:
            defn = SkillDefinition(
                name=name,
                category=category,
                aliases=aliases,
                requires_word_boundary=requires_boundary,
            )
            self._register(defn)

        # Add any extras
        if extra_skills:
            for defn in extra_skills:
                self._register(defn)

    def _register(self, defn: SkillDefinition) -> None:
        """Register a skill definition in the index."""
        self._skills.append(defn)

        # Index by canonical name (lowercase)
        self._lookup_index[defn.name.lower()] = defn

        # Index by each alias (lowercase)
        for alias in defn.aliases:
            self._lookup_index[alias.lower()] = defn

    def lookup(self, name: str) -> SkillDefinition | None:
        """
        Look up a skill by name or alias (case-insensitive).

        Args:
            name: Skill name or alias to look up.

        Returns:
            SkillDefinition if found, None otherwise.
        """
        return self._lookup_index.get(name.lower().strip())

    def get_all_skills(self) -> list[SkillDefinition]:
        """Return all registered skill definitions."""
        return list(self._skills)

    def get_by_category(self, category: SkillCategory) -> list[SkillDefinition]:
        """Return all skills in a specific category."""
        return [s for s in self._skills if s.category == category]

    def get_all_names_and_aliases(self) -> list[str]:
        """Return a flat list of all searchable names (canonical + aliases)."""
        return list(self._lookup_index.keys())

    def get_multi_word_entries(self) -> list[tuple[str, SkillDefinition]]:
        """
        Return all multi-word names/aliases sorted by length (longest first).

        Used by the pattern matcher to match longer phrases before shorter ones,
        e.g., "Ruby on Rails" before "Ruby".
        """
        entries = []
        for name_lower, defn in self._lookup_index.items():
            if " " in name_lower or "." in name_lower or "-" in name_lower:
                entries.append((name_lower, defn))
        entries.sort(key=lambda x: len(x[0]), reverse=True)
        return entries

    def get_single_word_entries(self) -> list[tuple[str, SkillDefinition]]:
        """
        Return all single-word names/aliases sorted by length (longest first).
        """
        entries = []
        seen = set()
        for name_lower, defn in self._lookup_index.items():
            if " " not in name_lower and "." not in name_lower and "-" not in name_lower:
                if name_lower not in seen:
                    entries.append((name_lower, defn))
                    seen.add(name_lower)
        entries.sort(key=lambda x: len(x[0]), reverse=True)
        return entries

    @property
    def size(self) -> int:
        """Return the number of unique skills (not counting aliases)."""
        return len(self._skills)

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[SkillDefinition]:
        return iter(self._skills)

    def __contains__(self, name: str) -> bool:
        return name.lower().strip() in self._lookup_index
