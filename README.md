# 🚀 NLP Transformers & RAG Platform

Welcome to the **NLP Transformers & RAG Platform**! This repository is an academic mini-project developed for **ENSA Al Hoceima (Ingénierie des Données - ID2)**. 

The project is structured sequentially in **two major phases**:
1. **Phase 1 (NLP Toolbox)**: Exploring pre-trained Hugging Face Transformers for text classification, sentiment analysis, extractive question answering, and text summarization.
2. **Phase 2 (Retrieval-Augmented Generation)**: Designing an intelligent, custom RAG system that processes academic slide decks and text documents, indexes them using vector embeddings, and enables a chatbot to answer queries based strictly on the provided documents.

---

## 🌟 Key Features

*   **Classic NLP Toolbox**: Implementation of modern Transformer architectures for text classification, sentiment polarity extraction, question answering over contexts, and text summarization.
*   **Knowledge Base Parsing**: Ingestion and semantic parsing of course slides (PDFs) and research texts (TXTs) with advanced chunking.
*   **Vector Search Engine**: Semantic context retrieval powered by local high-speed **FAISS** vector indexing and `sentence-transformers` embeddings.
*   **Response Synthesis**: Prompt engineering and generative LLM augmentation to deliver reliable, context-bounded answers.
*   **Interactive Web Application**: A unified, premium **Streamlit** dashboard with dedicated navigation tabs for classic NLP utilities and the RAG Chatbot.

---

## 📂 Project Directory Structure

The platform uses a modular, decoupled directory structure to allow clean development and deployment:

```text
.
├── data/
│   └── corpus/
│       └── README.md                       # Directory to store raw PDFs and TXT courses
├── notebooks/
│   ├── 01_exploration_transformers.ipynb   # Jupyter notebook for Phase 1 testing
│   └── 02_experimentation_rag.ipynb        # Jupyter notebook for RAG experimentation
├── src/
│   ├── __init__.py
│   ├── config.py                           # Centralized configuration (model tags, chunk sizes)
│   ├── core_nlp/
│   │   ├── __init__.py
│   │   └── nlp_pipelines.py                # Hugging Face Pipeline wrappers
│   ├── rag_engine/
│   │   ├── __init__.py
│   │   ├── document_loader.py              # Ingestion, PDF parsing, text splitter
│   │   ├── embedding_model.py              # Text embedder (SentenceTransformers)
│   │   └── vector_store.py                 # FAISS vector store manager (index construction)
│   └── generator/
│       ├── __init__.py
│       └── llm_generator.py                # Prompt templates & LLM generation caller
├── app.py                                  # Streamlit GUI with multi-tab structure
├── requirements.txt                        # Project dependencies
└── README.md                               # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/nlp-transformers-rag.git
cd nlp-transformers-rag
```

### 2. Configure a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📖 Reference Material & Credits

*   **Reference Book**: Inspired by *"Natural Language Processing with Transformers: Building Language Applications with Hugging Face"* (Chapters 2, 6, 7 & 9).
*   **Institution**: École Nationale des Sciences Appliquées - Al Hoceima (Ingénierie des Données - 2ème Année).
