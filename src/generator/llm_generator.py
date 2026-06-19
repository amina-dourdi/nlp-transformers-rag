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

        prompt = f"""Vous êtes un professeur. Répondez à la question en utilisant le contexte.

Contexte :
{combined_context}

---
Exemple :
Question : Qu'est-ce que le NLP ?
Réponse : Le NLP (Natural Language Processing) est une branche de l'IA qui permet aux ordinateurs de comprendre le langage humain.
---

Question : {query}
Réponse :"""
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
                max_new_tokens=100,
                do_sample=False
            )
            if not response or not response.strip():
                raise ValueError("Réponse API vide")
            return response.strip()

        except Exception as e:
            # 🚀 FALLBACK LOCAL : Exécution locale d'un petit LLM parlant français si l'API crashe
            try:
                from transformers import pipeline
                if not hasattr(self, "local_pipeline"):
                    try:
                        import streamlit as st
                        with st.spinner("L'API a échoué. Téléchargement d'un modèle d'IA local en français (Bloomz, ~1 Go)... Patientez la première fois !"):
                            self.local_pipeline = pipeline("text-generation", model=GENERATIVE_LLM_MODEL, device_map="auto")
                    except ImportError:
                        print("L'API a échoué. Téléchargement d'un modèle d'IA local en français (Bloomz, ~1 Go)... Patientez la première fois !")
                        self.local_pipeline = pipeline("text-generation", model=GENERATIVE_LLM_MODEL, device_map="auto")
                        
                res = self.local_pipeline(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
                return res[0]["generated_text"].strip()
            except Exception as e_local:
                return f"❌ L'API a échoué ({type(e).__name__}) ET le mode hors-ligne a échoué ({e_local})"

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
                max_new_tokens=100,
                do_sample=False
            )
            if not response or not response.strip():
                raise ValueError("Réponse API vide")
            return response.strip()

        except Exception as e:
            try:
                from transformers import pipeline
                if not hasattr(self, "local_pipeline"):
                    import streamlit as st
                    with st.spinner("L'API a échoué. Téléchargement d'un modèle d'IA local en français (Bloomz, ~1 Go)... Patientez la première fois !"):
                        self.local_pipeline = pipeline("text-generation", model="bigscience/bloomz-560m")
                
                res = self.local_pipeline(prompt, max_new_tokens=100, return_full_text=False)
                return res[0]["generated_text"].strip()
            except Exception as e_local:
                return f"❌ L'API a échoué ({type(e).__name__}) ET le mode hors-ligne a échoué ({e_local})"
