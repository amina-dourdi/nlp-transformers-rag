# src/config.py
import os

# Choix des modèles Hugging Face
NLP_CLASSIFIER_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
NLP_SUMMARIZER_MODEL = "facebook/bart-large-cnn"
NLP_QA_MODEL = "deepset/roberta-base-squad2"

# Choix des modèles RAG
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATIVE_LLM_MODEL = "google/gemma-2b-it"

# Hyperparamètres RAG
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
VECTOR_DB_PATH = "data/faiss_index"
CORPUS_DIR = "data/corpus"
