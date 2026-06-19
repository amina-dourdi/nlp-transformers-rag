# app.py
# ==========================================
# Interface Utilisateur Globale (Streamlit)
# ==========================================

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

# ==========================================
# ONGLET 1 : Boîte à outils NLP (Tâche A3)
# ==========================================
with tab1:
    st.header("Phase 1 : Exploration NLP")
    st.write("Testez les tâches NLP.")
    
    # Importer TOUTES les fonctions depuis nlp_pipelines
    from src.core_nlp.nlp_pipelines import (
        analyze_sentiment, 
        answer_question,
        classify_text,
        summarize_text
    )
    
    # Sous-onglets pour chaque tâche
    nlp_tab1, nlp_tab2, nlp_tab3, nlp_tab4 = st.tabs([
        "😊 Sentiment", "❓ Question Answering", "🏷️ Classification", "📝 Résumé"
    ])
    
    # ── 1. Analyse de Sentiment ──
    with nlp_tab1:
        st.subheader("Analyse de Sentiment")
        st.write("Déterminez si un texte est **positif** ou **négatif**.")
        
        sentiment_input = st.text_area(
            "Entrez votre texte :", 
            value="I love learning about NLP and Transformers!",
            key="sentiment_input"
        )
        
        if st.button("🔍 Analyser le Sentiment", key="btn_sentiment"):
            with st.spinner("Analyse en cours..."):
                result = analyze_sentiment(sentiment_input)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Label", result["label"])
            with col2:
                st.metric("Confiance", f"{result['score']:.2%}")
                
    # ── 2. Question Answering ──
    with nlp_tab2:
        st.subheader("Question Answering")
        st.write("Posez une question et fournissez un contexte. Le modèle extraira la réponse du contexte.")
        
        qa_context = st.text_area(
            "Contexte :",
            value="Le NLP est une branche de l'intelligence artificielle. Les Transformers ont été introduits en 2017 par Google.",
            height=150,
            key="qa_context"
        )
        qa_question = st.text_input(
            "Question :", 
            value="Quand les Transformers ont-ils été introduits ?",
            key="qa_question"
        )
        
        if st.button("❓ Trouver la Réponse", key="btn_qa"):
            with st.spinner("Recherche de la réponse..."):
                result = answer_question(qa_question, qa_context)
            
            st.success(f"**Réponse :** {result['answer']}")
            st.write(f"Score de confiance : {result['score']:.2f}")

    # ── 3. Classification ──
    with nlp_tab3:
        st.subheader("Classification de Texte (Émotions)")
        classif_input = st.text_area(
            "Entrez votre texte :", 
            value="I am so excited about this project!",
            key="classif_input"
        )
        if st.button("🏷️ Classifier", key="btn_classif"):
            with st.spinner("Classification en cours..."):
                result = classify_text(classif_input)
            
            # Gestion du format de retour de la fonction de Firdawss
            if "status" in result and result["status"] == "success":
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Émotion", result["label"])
                with col2:
                    st.metric("Confiance", f"{result['confidence']:.2%}")
            else:
                st.error("Erreur lors de la classification.")
                    
    # ── 4. Résumé Automatique ──
    with nlp_tab4:
        st.subheader("Résumé Automatique")
        
        long_text = """The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time not articulated as a problem separate from artificial intelligence (AI).
        
The premise of symbolic NLP is well-summarized by John Searle's Chinese room experiment: Given a collection of rules (e.g., a Chinese phrasebook, with questions and matching answers), the computer emulates natural language understanding (or other NLP tasks) by applying those rules to the data it confronts.

Up to the 1980s, most natural language processing systems were based on complex sets of hand-written rules. Starting in the late 1980s, however, there was a revolution in natural language processing with the introduction of machine learning algorithms for language processing. This was due to both the steady increase in computational power (see Moore's law) and the gradual lessening of the dominance of Chomskyan theories of linguistics (e.g. transformational grammar), whose theoretical underpinnings discouraged the sort of corpus linguistics that is underlying the machine-learning approach to language processing."""
        
        summary_input = st.text_area(
            "Texte à résumer :",
            value=long_text,
            height=300,
            key="summary_input"
        )
        
        if st.button("📝 Résumer", key="btn_summary"):
            with st.spinner("Résumé en cours..."):
                result = summarize_text(summary_input)
            st.success("**Résumé :**")
            st.write(result)


# ==========================================
# ONGLET 2 : Chatbot RAG Intelligent (Phase 2)
# ==========================================
with tab2:
    st.header("Phase 2 : Système RAG")
    st.write("Posez des questions sur vos cours stockés dans `data/corpus/`.")
    
    # Imports RAG
    try:
        from src.rag_engine.document_loader import load_pdf_documents, get_chunks
        from src.rag_engine.vector_store import VectorStoreManager
        firdawss_rag_ready = True
    except ImportError:
        firdawss_rag_ready = False
        
    from src.generator.llm_generator import RAGGenerator
    from src.config import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH, CORPUS_DIR

    if not firdawss_rag_ready:
        st.warning("⏳ En attente de l'implémentation par Firdawss (Tâches T5 à T9) pour le RAG.")
    else:
        # Initialisation des gestionnaires dans la session (pour ne pas tout recharger)
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = VectorStoreManager(EMBEDDING_MODEL_NAME, VECTOR_DB_PATH)
        # Forcer le rechargement du générateur pour appliquer nos changements
        st.session_state.rag_generator = RAGGenerator()
        if "is_indexed" not in st.session_state:
            st.session_state.is_indexed = False

        col1, col2 = st.columns([1, 2])
        
        # Colonne 1 : Indexation (Tâche de Firdawss)
        with col1:
            st.subheader("1. Indexation")
            st.info("Avant de poser une question, il faut lire les PDF et créer la base de connaissances (VectorStore).")
            
            if st.button("🔄 Indexer les Documents PDF", key="btn_index"):
                with st.spinner("Lecture des PDF et création des embeddings..."):
                    # 1. Charger PDF
                    docs = load_pdf_documents(CORPUS_DIR)
                    if not docs:
                        st.error("Aucun PDF trouvé ou le code de lecture (T5) n'est pas encore prêt.")
                    else:
                        # 2. Découper en morceaux
                        chunks = get_chunks(docs)
                        if not chunks:
                            st.error("Le découpage en chunks (T6) n'est pas encore prêt.")
                        else:
                            # 3. Sauvegarder dans FAISS
                            st.session_state.vector_store.build_and_save_index(chunks)
                            st.session_state.is_indexed = True
                            st.success(f"✅ Terminé ! {len(docs)} PDF lus et découpés en {len(chunks)} morceaux.")
                        
        # Colonne 2 : Chatbot (Tâche d'Amina)
        with col2:
            st.subheader("2. Chatbot RAG")
            if not st.session_state.is_indexed:
                st.warning("⚠️ Veuillez d'abord indexer les documents avec le bouton à gauche.")
            else:
                rag_query = st.text_input("Posez votre question au LLM :", value="Qu'est-ce qu'un système RAG ?")
                
                if st.button("🚀 Obtenir la Réponse"):
                    with st.spinner("Recherche dans les PDF et Génération LLM..."):
                        try:
                            # ÉTAPE 1 : Firdawss cherche le contexte
                            contexts = st.session_state.vector_store.search_top_k(rag_query, k=3)
                            
                            # ÉTAPE 2 : Amina génère la réponse
                            result = st.session_state.rag_generator.generate(rag_query, contexts)
                            
                            st.success("**Réponse du Modèle :**")
                            st.info(result)
                            
                            # Afficher les sources trouvées
                            with st.expander("🔍 Voir les sources utilisées par le modèle"):
                                for i, ctx in enumerate(contexts):
                                    st.markdown(f"**Source {i+1} : {ctx['source']}**")
                                    st.info(ctx['text'])
                                    
                        except Exception as e:
                            st.error(f"Une erreur est survenue lors de la génération : {e}")
