# src/generator/llm_generator.py
# ==========================================
# PHASE 2 : RAG - Générateur LLM
# ==========================================

#  API HuggingFace Inference
# Pour obtenir un token gratuit :
#   1. Créer un compte sur https://huggingface.co
#   2. Settings → Access Tokens → New Token
#   3. Définir la variable d'environnement HF_TOKEN
#      ou remplacer directement dans le code ci-dessous.

import os
from typing import List, Dict
from huggingface_hub import InferenceClient
from src.config import GENERATIVE_LLM_MODEL


class RAGGenerator:
    def __init__(self, model_name: str = GENERATIVE_LLM_MODEL, token: str = None):
        """
        Initialise le client d'inférence HuggingFace.

        Args:
            model_name (str): Nom du modèle sur HuggingFace Hub.
            token (str): Token d'API HuggingFace. Si None, lit la variable
                         d'environnement HF_TOKEN.
        """
        # Récupérer le token : paramètre > variable d'environnement
        self.token = token or os.environ.get("HF_TOKEN", "")

        if not self.token:
            print(
                "⚠️  Aucun token HuggingFace détecté !\n"
                "    → Définissez la variable d'environnement HF_TOKEN\n"
                "    → Ou passez le token en paramètre : RAGGenerator(token='hf_...')"
            )

        self.model_name = model_name
        self.client = InferenceClient(
            model=self.model_name,
            token=self.token,
        )
        print(f"✅ Générateur LLM connecté au modèle '{self.model_name}'")

    # ──────────────────────────────────────────────
    # Construction du Prompt Augmenté (RAG)
    # ──────────────────────────────────────────────

    def build_prompt(self, query: str, contexts: List[Dict[str, str]]) -> str:
        """
        Formate le prompt augmenté avec les contextes trouvés par Firdawss.

        Le prompt est structuré pour :
        1. Donner un rôle clair au LLM (assistant pédagogique).
        2. Lui fournir le contexte extrait des documents (chunks FAISS).
        3. Lui interdire d'inventer des informations (garde-fou anti-hallucination).
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

    # ──────────────────────────────────────────────
    # Génération AVEC RAG (contexte fourni)
    # ──────────────────────────────────────────────

    def generate(self, query: str, contexts: List[Dict[str, str]]) -> str:
        """
        Assemble le prompt augmenté et génère la réponse via l'API HuggingFace.

        Args:
            query (str): La question de l'utilisateur.
            contexts (List[Dict]): Les chunks pertinents trouvés par FAISS.
                                   Chaque dict contient 'text' et 'source'.

        Returns:
            str: La réponse générée par le LLM.
        """
        prompt = self.build_prompt(query, contexts)

        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=512,      # Longueur maximale de la réponse générée
                temperature=0.3,         # Peu d'aléatoire → réponses précises et fiables
                repetition_penalty=1.2,  # Pénalise les répétitions dans le texte généré
            )
            return response.strip()

        except Exception as e:
            return f"❌ Erreur de génération : {e}"

    # ──────────────────────────────────────────────
    # Génération SANS RAG (pour la comparaison)
    # ──────────────────────────────────────────────

    def generate_without_rag(self, query: str) -> str:
        """
        Génère une réponse SANS contexte RAG.

        Utilisé pour la démonstration comparative :
        on montre que sans les documents, le LLM peut halluciner
        ou donner des réponses vagues / incorrectes.

        Args:
            query (str): La question de l'utilisateur.

        Returns:
            str: La réponse générée sans contexte.
        """
        prompt = f"""Répondez à la question suivante de manière claire et détaillée :

Question : {query}

Réponse :"""

        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=512,
                temperature=0.3,
                repetition_penalty=1.2,
            )
            return response.strip()

        except Exception as e:
            return f"❌ Erreur de génération : {e}"
