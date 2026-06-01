# src/core_nlp/nlp_pipelines.py
# ==========================================
# PHASE 1 : exploration et pipelines NLP
# ==========================================

# 🟦 Amina's Tasks:
# 1. Analyse de sentiment (sentiment-analysis)
# 2. Question Answering (QA) - extraction de réponse

# 🟫 Firdawss's Tasks:
# 1. Classification de texte (text-classification)
# 2. Résumé automatique (summarization)

from transformers import AutoTokenizer, AutoModel
from src.config import *

def load_pretrained_model_and_tokenizer(model_name: str):
    """
    Charge un modèle pré-entraîné et son tokenizer associé depuis Hugging Face.
    
    Code basé sur le Livre de Référence (Chapitre 2) :
    - AutoTokenizer.from_pretrained() -> Page 27
    - AutoModel.from_pretrained()     -> Page 32
    """
    print(f"🔄 Chargement du tokenizer pour '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"🔄 Chargement du modèle pré-entraîné '{model_name}'...")
    model = AutoModel.from_pretrained(model_name)
    
    print("✅ Modèle et Tokenizer chargés avec succès !")
    return tokenizer, model

def init_pipelines():
    """
    Initialiser les pipelines requis.
    """
    pass
