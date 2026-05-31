# app.py
# ==========================================
# Interface Utilisateur Globale (Streamlit)
# ==========================================

# 🟦 Amina's Task:
# - Créer l'interface Web interactive complète
# - Onglet 1 : Boîte à outils NLP (Phase 1)
# - Onglet 2 : Chatbot RAG Intelligent (Phase 2)
# - Comparaison de réponses avec / sans RAG

import streamlit as st
import os

st.set_page_config(
    page_title="ENSA NLP & RAG Tool",
    page_icon="🚀",
    layout="wide"
)

st.title("📚 Mini-Projet : Transformers & Systèmes RAG")
st.write("Bienvenue dans votre plateforme d'apprentissage NLP interactive.")

# Menu de navigation par onglets
tab1, tab2 = st.tabs(["⚙️ Boîte à outils NLP (Phase 1)", "💬 Chatbot RAG Intelligent (Phase 2)"])

with tab1:
    st.header("Phase 1 : Exploration NLP")
    st.write("Testez les tâches NLP implémentées par Amina et Firdawss.")
    
    # 1. Classification de texte (Firdawss)
    # 2. Analyse de sentiment (Amina)
    # 3. Question Answering (Amina)
    # 4. Résumé automatique (Firdawss)

with tab2:
    st.header("Phase 2 : Système RAG")
    st.write("Posez des questions sur vos cours stockés dans `data/corpus/`.")
    
    # Intégration du VectorStore (Firdawss) et de la Génération (Amina)
