"""Sample resumes and job descriptions for ResumeLens AI demonstration and testing."""

RESUME_BULLET_POINT_STRENGTH = """ALEX CHEN
San Jose, CA | alex.chen@email.com | github.com/alexchen-cs | linkedin.com/in/alexchen-dev

EDUCATION
University of California, Davis
Bachelor of Science in Computer Science, Minor in Statistics
Expected Graduation: May 2025
Relevant Coursework: Data Structures & Algorithms, Machine Learning, Artificial Intelligence, Database Systems, Linear Algebra

SKILLS
- Languages: Python, C++, Java, SQL, JavaScript
- Frameworks & Libraries: PyTorch, scikit-learn, Pandas, NumPy, Matplotlib, OpenCV
- Developer Tools: Git, GitHub, Linux/Unix, Docker, Jupyter Notebook, VS Code

EXPERIENCE
Undergraduate Machine Learning Research Assistant
Davis Machine Learning & Vision Lab | Oct 2023 - Present
- Worked with machine learning models to classify image datasets using convolutional neural networks.
- Helped analyze experimental data and cleaned image inputs for the team's computer vision benchmark pipeline.
- Used Python scripts to automate model training loops and saved checkpoint weights.
- Assisted graduate students with hyperparameter tuning experiments and recorded validation accuracy results.

Software Engineering Intern
Nexora Technologies | Jun 2023 - Sep 2023
- Assisted backend engineering team with API development and bug fixes for the customer dashboard.
- Wrote unit tests in Python using PyTest to increase code coverage across user authentication endpoints.
- Attended daily standups and collaborated with product managers to implement feature requests.
- Created documentation in Notion detailing the onboarding setup for internal microservices.

PROJECTS
Multimodal Sentiment Analysis Pipeline | PyTorch, Hugging Face, Python
- Developed a multimodal sentiment classifier using text and audio embeddings to predict emotional polarity.
- Cleaned and preprocessed over 20,000 conversational speech clips using Librosa and Pandas.
- Implemented baseline evaluation comparisons across ResNet and DistilBERT architectures.

Distributed Key-Value Store | C++, POSIX Threads, Sockets
- Built a multi-threaded, persistent key-value store supporting concurrent GET and SET requests.
- Handled network communications using TCP socket programming and implemented mutual exclusion locks to prevent race conditions.
- Tested throughput across varying concurrency levels using an automated benchmark client."""

RESUME_JOB_MATCH = """JORDAN TAYLOR
Boston, MA | jtaylor@alum.mit.edu | github.com/jordantaylor-ml | linkedin.com/in/jordant-ai

EDUCATION
Northeastern University | Boston, MA
Master of Science in Computer Science (Concentration in Machine Learning) | GPA: 3.85 / 4.0
Graduation: May 2024

Worcester Polytechnic Institute (WPI) | Worcester, MA
Bachelor of Science in Data Science & Applied Mathematics | GPA: 3.80 / 4.0
Graduation: May 2022

TECHNICAL SKILLS
- Programming Languages: Python, C++, SQL (PostgreSQL), Bash, R
- ML & Deep Learning: PyTorch, Hugging Face Transformers, scikit-learn, XGBoost, TorchVision, NumPy, Pandas, SciPy
- MLOps & Experiment Tracking: Weights & Biases (W&B), MLflow, Git, GitHub Actions, Linux
- Deployment & Web Services: FastAPI, Flask, RESTful APIs, ONNX Runtime, Redis

PROFESSIONAL & RESEARCH EXPERIENCE
Graduate Machine Learning Researcher
Khoury College NLP & Information Retrieval Group | Sep 2022 - May 2024
- Fine-tuned transformer-based models (RoBERTa, DeBERTa-v3) on 500k+ domain-specific clinical text records for named entity recognition (NER), achieving a 4.2% F1-score improvement over existing baselines.
- Profiled GPU memory bottlenecks and implemented distributed data parallel (DDP) training across multi-node NVIDIA A100 clusters, reducing epoch training time by 35%.
- Implemented comprehensive evaluation suites measuring precision, recall, inference latency, and calibration error across out-of-distribution test sets.
- Tracked hyperparameter sweeps and model artifact versioning for 60+ experimental runs using Weights & Biases (W&B).

Machine Learning Intern
Acuity Financial Technologies | Jun 2023 - Aug 2023
- Built an end-to-end fraud detection pipeline using XGBoost and Random Forests on tabular transaction logs, increasing precision from 0.82 to 0.89 on held-out test data.
- Deployed trained inference models as low-latency microservices with FastAPI and ONNX Runtime, maintaining sub-30ms p99 response times under load.
- Designed automated data validation scripts with Pandas and Great Expectations to detect feature drift and missing values in incoming data streams.
- Presented technical benchmarks and model interpretability findings (SHAP values) to senior engineering leadership.

PROJECTS
Neural Semantic Search & Retrieval System | PyTorch, Hugging Face, ChromaDB, FastAPI
- Designed a bi-encoder neural retrieval pipeline combining dense vector embeddings (sentence-transformers) with BM25 keyword search using Reciprocal Rank Fusion (RRF).
- Indexed 150,000 arXiv scientific abstracts into ChromaDB vector database, enabling sub-second hybrid semantic queries.
- Built a demonstration web UI and REST API with FastAPI for real-time query retrieval and reranking.

Distributed Edge Vision Classifier | PyTorch, OpenCV, TensorRT, C++
- Quantized ResNet and MobileNet models to INT8 precision using TensorRT, achieving a 2.8x speedup on edge hardware with less than 1% drop in top-1 accuracy.
- Benchmarked frame processing pipelines with OpenCV and multithreaded C++ queues for real-time video stream ingestion."""

JOB_DESCRIPTION_ML_ENGINEER = """Machine Learning Engineer (Early Career / New Grad)
Apex AI Labs - Boston, MA (Hybrid)

About the Role:
Apex AI Labs is seeking an energetic Machine Learning Engineer to join our Applied AI Core team. In this role, you will design, train, and deploy machine learning models that power our enterprise intelligence platform. You will collaborate closely with data engineering and platform teams to take models from research prototypes to high-scale production systems.

Key Responsibilities:
- Develop, evaluate, and optimize deep learning and natural language processing (NLP) models for production use cases.
- Build resilient data preprocessing, feature engineering, and model inference pipelines.
- Containerize ML services and deploy scalable inference endpoints on cloud infrastructure (AWS / GCP).
- Implement automated model evaluation, monitoring, and experiment tracking to maintain high performance in production.
- Collaborate with software engineers to integrate ML models into core backend microservices.

Requirements & Qualifications:
- M.S. or B.S. in Computer Science, Data Science, Electrical Engineering, or related quantitative field.
- Strong proficiency in Python and modern deep learning frameworks (PyTorch or TensorFlow).
- Hands-on experience with NLP architectures (Transformers, BERT, LLM fine-tuning) or Computer Vision models.
- Demonstrated experience deploying models via REST APIs (e.g., FastAPI, Flask, or Triton Inference Server).
- Solid software engineering fundamentals: Git version control, unit testing, and clean modular code design.
- Familiarity with MLOps and experiment tracking tools such as Weights & Biases, MLflow, or TensorBoard.

Preferred Qualifications:
- Experience with containerization technologies (Docker) and container orchestration (Kubernetes).
- Hands-on experience with cloud platforms (AWS services such as SageMaker, S3, EC2, or GCP Vertex AI).
- Experience with vector search databases (Pinecone, Milvus, Qdrant, or ChromaDB) and retrieval-augmented generation (RAG).
- Knowledge of distributed model training (PyTorch DDP, DeepSpeed, or Ray).
- Experience with workflow orchestration engines like Apache Airflow or Kubeflow."""

EXAMPLES = [
    [
        RESUME_BULLET_POINT_STRENGTH,
        "Bullet Point Strength",
        "",
    ],
    [
        RESUME_JOB_MATCH,
        "Job Description Match",
        JOB_DESCRIPTION_ML_ENGINEER,
    ],
]
