# src/config.py
import os

# Choix des modèles Hugging Face (Phase 1)
NLP_CLASSIFIER_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion" # Classification d'émotions (Chapitre 2 du livre)
NLP_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"   # Analyse de sentiment (Positif/Négatif)
NLP_QA_MODEL = "deepset/roberta-base-squad2"                               # Question Answering (QA)
NLP_SUMMARIZER_MODEL = "facebook/bart-large-cnn"                           # Résumé automatique (Summarization)

# Choix des modèles RAG
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATIVE_LLM_MODEL = "google/gemma-2b-it"

# Hyperparamètres RAG
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
VECTOR_DB_PATH = "data/faiss_index"
CORPUS_DIR = "data/corpus"
