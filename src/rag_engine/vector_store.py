# src/rag_engine/vector_store.py
# ==========================================
# PHASE 2 : RAG - Indexation Vectorielle FAISS
# ==========================================

# 🟫 Firdawss's Tasks:
# - Gérer la base vectorielle FAISS (création, indexation, recherche top-k)

from typing import List, Dict
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStoreManager:
    def __init__(self, model_name: str, index_path: str):
        self.encoder = SentenceTransformer(model_name)
        self.index_path = index_path
        self.index = None
        self.documents = []  # Liste de dict : [{"id": int, "text": str, "source": str}]

    def build_and_save_index(self, chunks: List[Dict[str, str]]):
        """
        Reçoit les chunks, calcule les embeddings et initialise FAISS.
        """
        self.documents = chunks
        texts = [c["text"] for c in chunks]
        
        # 1. Calculer les embeddings
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        
        # 2. Créer l'index FAISS
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # 3. Sauvegarder l'index
        print(f"✅ Index FAISS créé avec {self.index.ntotal} documents.")

    def search_top_k(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """
        Encode la requête et retourne les K documents les plus proches.
        """
        if self.index is None:
            raise ValueError("L'index FAISS n'est pas initialisé ou chargé !")
            
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results
