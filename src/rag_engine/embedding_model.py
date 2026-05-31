# src/rag_engine/embedding_model.py
# ==========================================
# PHASE 2 : RAG - Embeddings sémantiques
# ==========================================

# 🟫 Firdawss's Tasks:
# - Encoder sémantiquement les segments textuels en vecteurs numériques.

from typing import List
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

class EmbeddingModelWrapper:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Calculer les vecteurs pour une liste de textes.
        """
        return self.model.encode(texts, show_progress_bar=True).tolist()
