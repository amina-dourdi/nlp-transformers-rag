# src/config.py
import os

# General and safe models have been selected to run on any device
NLP_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
NLP_QA_MODEL = "deepset/roberta-base-squad2"
NLP_CLASSIFIER_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"
NLP_SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
                    
# Choix des modèles RAG
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATIVE_LLM_MODEL = "google/gemma-2b-it"

# Hyperparamètres RAG
CHUNK_SIZE = 600       # Size of each chunk (in characters) to preserve context accuracy
CHUNK_OVERLAP = 100    # Overlap between chunks to avoid splitting ideas and sentences

VECTOR_DB_PATH = "data/faiss_index"
CORPUS_DIR = "data/corpus"
