# src/core_nlp/nlp_pipelines.py
# ==========================================
# PHASE 1 : exploration et pipelines NLP
# ==========================================

# 1. Analyse de sentiment (sentiment-analysis)
# 2. Question Answering (QA) - extraction de réponse

# 3. Classification de texte (text-classification)
# 4. Résumé automatique (summarization)

from transformers import AutoTokenizer, pipeline, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
from src.config import *
import torch


# ============================================================================
# GLOBAL CACHES INITIALIZATION
# ============================================================================
_sentiment_pipeline = None
_qa_tokenizer = None
_qa_model = None
_classification_pipeline = None

# Summarization is loaded explicitly and reliably to prevent Pipeline Registry errors
_summarization_tokenizer = None
_summarization_model = None

def init_pipelines():
    """
    [Task F3 - Corrected] Initialize all Hugging Face models used in the project.
    Fixes the Hugging Face KeyError by explicitly loading the Seq2Seq architecture for BART.
    """
    global _sentiment_pipeline, _qa_tokenizer, _qa_model
    global _classification_pipeline, _summarization_tokenizer, _summarization_model
    
    print("==================================================")
    print("🔄 INITIALIZATION OF ALL PROJECT MODEL CACHES...")
    print("==================================================")
    
    # ────────────────────────────────────
    print("🔄 Initializing Sentiment Analysis Pipeline...")
    _sentiment_pipeline = pipeline("sentiment-analysis", model=NLP_SENTIMENT_MODEL)
    
    # ────────────────────────────────────
    print("🔄 Initializing Question Answering Components (Manual Setup)...")
    _qa_tokenizer = AutoTokenizer.from_pretrained(NLP_QA_MODEL)
    _qa_model = AutoModelForQuestionAnswering.from_pretrained(NLP_QA_MODEL)
    
    # ────────────────────────────────────
    print("🔄 Initializing Text Classification Pipeline...")
    _classification_pipeline = pipeline("text-classification", model=NLP_CLASSIFIER_MODEL)

    # ────────────────────────────────────
    print("🔄 Initializing Explicit Text Summarization Components...")
    # Explicitly load the encoder-decoder architecture specialized for summarization
    # to avoid issues in recent versions of the transformers library
    _summarization_tokenizer = AutoTokenizer.from_pretrained(NLP_SUMMARIZATION_MODEL)
    _summarization_model = AutoModelForSeq2SeqLM.from_pretrained(NLP_SUMMARIZATION_MODEL)
    
    print("\n✅ All pipelines are successfully cached and ready without errors!")

# ============================================================================


def analyze_sentiment(text: str) -> dict:
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        init_pipelines()
    result = _sentiment_pipeline(text)
    return result[0]

# ============================================================================

def answer_question(question: str, context: str) -> dict:
    global _qa_tokenizer, _qa_model
    if _qa_tokenizer is None or _qa_model is None:
        init_pipelines()
        
    inputs = _qa_tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = _qa_model(**inputs)
    
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    
    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    answer = _qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    score = (torch.max(outputs.start_logits) + torch.max(outputs.end_logits)).item() / 2.0
    return {"answer": answer, "score": score}

# ============================================================================

def classify_text(text: str) -> dict:
    global _classification_pipeline
    if _classification_pipeline is None:
        init_pipelines()
    try:
        prediction = _classification_pipeline(text)[0]
        return {
            "status": "success",
            "label": prediction["label"],
            "confidence": round(prediction["score"], 4)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================================================

def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> dict:
    """Generates a summary using an explicit Seq2Seq generation loop to guarantee execution."""
    global _summarization_tokenizer, _summarization_model
    if _summarization_tokenizer is None or _summarization_model is None:
        init_pipelines()
    try:
        # Convert the input text into token IDs for the BART model
        inputs = _summarization_tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
        
        # Generation loop following the recommended settings from the book
        summary_ids = _summarization_model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            early_stopping=True
        )
        
        # Decode and return the final generated summary
        summary_text = _summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return {"summary_text": summary_text}
    except Exception as e:
        return {"summary_text": f"Error during summarization: {str(e)}"}