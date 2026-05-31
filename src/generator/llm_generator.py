# src/generator/llm_generator.py
# ==========================================
# PHASE 2 : RAG - Générateur LLM
# ==========================================

# 🟦 Amina's Tasks:
# - Prompt engineering : construction du prompt augmenté
# - Génération de réponses à partir de modèles ou API

from typing import List, Dict
from src.config import GENERATIVE_LLM_MODEL

class RAGGenerator:
    def __init__(self, model_name_or_api_key: str = GENERATIVE_LLM_MODEL):
        # Amina : Initialiser le modèle local Hugging Face ou un client API
        pass

    def build_prompt(self, query: str, contexts: List[Dict[str, str]]) -> str:
        """
        Formate le prompt augmenté avec les contextes trouvés par Firdawss.
        """
        combined_context = "\n\n".join([
            f"[Source: {c['source']}]\n{c['text']}" 
            for c in contexts
        ])
        
        prompt = f"""Vous êtes un assistant IA pédagogique et rigoureux.
Répondez de manière structurée et claire à la question posée en vous basant uniquement sur le contexte ci-dessous.
Si la réponse ne figure pas dans le contexte fourni, dites simplement : "Je ne trouve pas la réponse dans les documents fournis."

Contexte extrait du cours :
---------------------
{combined_context}
---------------------

Question de l'étudiant : {query}

Réponse claire et détaillée :"""
        return prompt

    def generate(self, query: str, contexts: List[Dict[str, str]]) -> str:
        """
        Assemble le prompt et génère le texte de réponse final.
        """
        prompt = self.build_prompt(query, contexts)
        
        # Amina : Appeler le LLM et renvoyer la réponse générée
        return "Réponse générée (à connecter au modèle LLM d'Amina)"
