# Content Search Application

This is a search application that implements three different search relevance models:

- **TF-IDF Model:** Uses TF-IDF vectorization to compute text similarity.
- **Sentence Transformer Model:** Uses state-of-the-art Sentence Transformers for semantic similarity.
- **Hybrid Model:** Combines TF-IDF and Sentence Transformer approaches to boost search performance.
- **Advanced Retrieval (FAISS + RAG + LLM):** Leverages FAISS for efficient indexing, and Retrieval-Augmented Generation (RAG) with an LLM in combination with SentenceTransformer embeddings to generate natural language recommendations for content retrieval.


This repository follows a modular design. Common functionalities (data loading, preprocessing, evaluation, recommendation generation, and logging) are encapsulated in classes within the `components` folder, while each search model has its own implementation in the `utils` folder.


## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/UKEYBHAKTI002922939/bhakti_study.git
   cd bhakti_study

2. **Create and Activate a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate


3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt

**Alternatively, install the package in editable mode:**

    pip install -e .


## Usage

**Run the search application using the job_runner.py script. Use the --model flag to select the model:**
### TF-IDF Model:

    python job_runner.py --model tfidf

### Sentence Transformer Model:
   python job_runner.py --model sentence

### Hybrid Model:

   python job_runner.py --model hybrid

Each run will:

- Load raw data from data/raw.
- Preprocess the data (creating columns like processed_title).
- Execute the chosen search model.
- Evaluate model performance (logging metrics such as F1, accuracy, precision, recall).
- Generate and log recommendations.
- Write log files (with detailed metric outputs) to the logs folder.

## Logging
Logs are automatically created in the logs directory. Each run generates a timestamped log file (e.g., log_2025-03-15-14-30-00.log) containing detailed information, including:

- Data loading and preprocessing steps.
- Model threshold tuning and evaluation metrics.
- Classification reports with metrics (F1, precision, recall, etc.).

