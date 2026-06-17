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

from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForQuestionAnswering
from src.config import *
import torch

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

# Initialisation globale (mise en cache) pour éviter de recharger les modèles à chaque appel
_sentiment_pipeline = None
_qa_tokenizer = None
_qa_model = None

_classification_pipeline = None
_summarizer_tokenizer = None
_summarizer_model = None


def init_pipelines():
    """
    Initialise les modèles Hugging Face pour les tâches d'Amina.
    """
    global _sentiment_pipeline, _qa_tokenizer, _qa_model, _classification_pipeline
    global _summarizer_tokenizer, _summarizer_model
    
   

    print("🔄 Initialisation du pipeline d'Analyse de Sentiment...")
    try:
        _sentiment_pipeline = pipeline("sentiment-analysis", model=NLP_SENTIMENT_MODEL)
    except Exception as e:
        print(f"Erreur Sentiment: {e}")
    
    print("🔄 Initialisation des composants de Question Answering (Manuel)...")
    try:
        _qa_tokenizer = AutoTokenizer.from_pretrained(NLP_QA_MODEL)
        _qa_model = AutoModelForQuestionAnswering.from_pretrained(NLP_QA_MODEL)
    except Exception as e:
        print(f"Erreur QA: {e}")
    
    print("🔄 Initializing Text Classification Pipeline ...")
    try:
        _classification_pipeline = pipeline("text-classification", model=NLP_CLASSIFICATION_MODEL)
    except Exception as e:
        print(f"Erreur Classification: {e}")

    print("🔄 Initializing Text Summarization Pipeline (Manuel)...")
    try:
        from transformers import AutoModelForSeq2SeqLM
        _summarizer_tokenizer = AutoTokenizer.from_pretrained(NLP_SUMMARIZATION_MODEL)
        _summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(NLP_SUMMARIZATION_MODEL)
    except Exception as e:
        print(f"Erreur Summarization: {e}")
        
    print("✅ Initialisation terminée (avec ou sans erreurs) !")

def analyze_sentiment(text: str) -> dict:
    """
    Fonction pour l'analyse de sentiment (Tâche Amina).
    """
    if _sentiment_pipeline is None:
        init_pipelines()
    
    # Le pipeline retourne une liste de dictionnaires, ex: [{'label': 'POSITIVE', 'score': 0.99}]
    result = _sentiment_pipeline(text)
    return result[0]

def answer_question(question: str, context: str) -> dict:
    """
    Fonction pour le Question Answering (Tâche Amina).
    Implémentation manuelle basée sur le Chapitre 7 du livre.
    """
    if _qa_tokenizer is None or _qa_model is None:
        init_pipelines()
        
    # 1. Tokenisation de la question et du contexte
    inputs = _qa_tokenizer(question, context, return_tensors="pt")
    
    # 2. Passage dans le modèle pour obtenir les logits de début et de fin
    with torch.no_grad():
        outputs = _qa_model(**inputs)
    
    # 3. Trouver les indices avec le score le plus élevé
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    
    # 4. Extraire les tokens de la réponse et décoder en texte
    answer_tokens = inputs.input_ids[0][start_idx:end_idx]
    answer = _qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    # Calculer un score de confiance basique (optionnel, pour garder le format)
    score = (torch.max(outputs.start_logits) + torch.max(outputs.end_logits)).item() / 2.0
    
    return {"answer": answer, "score": score}

def classify_text(text: str) -> dict:
    """
    Executes raw text classification using the cached pipeline.
    
    Args:
        text (str): The raw text sequence to be evaluated (e.g., product review or statement).
        
    Returns:
        dict: A formatted dictionary tracking the predicted label string and confidence float.
    """
    if _classification_pipeline is None:
        init_pipelines()
        
    try:
        # Run inference through the pipeline layer
        prediction = _classification_pipeline(text)[0]
        
        return {
            "status": "success",
            "label": prediction["label"],
            "confidence": round(prediction["score"], 4)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    


def summarize_text(text: str) -> str:
    """
    Generates a concise abstractive summary from a raw dialogue or text sequence 
    using the configured manual model.
    """
    global _summarizer_tokenizer, _summarizer_model
    
    if _summarizer_model is None:
        init_pipelines()
        
    try:
        if _summarizer_model is None:
            return "Erreur : Le modèle de résumé n'a pas pu être chargé correctement (environnement Python)."
            
        # Tokenisation
        inputs = _summarizer_tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
        
        # Génération
        summary_ids = _summarizer_model.generate(
            inputs["input_ids"],
            length_penalty=0.8,
            num_beams=8,
            max_length=128,
            min_length=32
        )
        
        # Décodage
        summary = _summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        print(f"❌ Generation Error: {str(e)}")
        return f"Error during summarization: {str(e)}"