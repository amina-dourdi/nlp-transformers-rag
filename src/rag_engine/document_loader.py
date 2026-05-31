# src/rag_engine/document_loader.py
# ==========================================
# PHASE 2 : RAG - Chargement & Découpage
# ==========================================

# 🟫 Firdawss's Tasks:
# - Charger les documents PDF ou TXT
# - Découper les textes en chunks (segments)

from typing import List, Dict
import os
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf_documents(folder_path: str) -> List[Dict[str, str]]:
    """
    Charger tous les documents PDF du dossier corpus.
    Retourne une liste de dict : [{"text": "...", "source": "nom_fichier.pdf"}]
    """
    pass

def get_chunks(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Découper les documents chargés en chunks homogènes.
    """
    pass
