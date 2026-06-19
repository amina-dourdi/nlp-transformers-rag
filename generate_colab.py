import json
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
notebook_path = os.path.join(project_dir, "Final_Project_Colab.ipynb")

def read_file(filepath):
    with open(os.path.join(project_dir, filepath), 'r', encoding='utf-8') as f:
        return f.read()

try:
    config_code = read_file("src/config.py")
    nlp_code = read_file("src/core_nlp/nlp_pipelines.py")
    loader_code = read_file("src/rag_engine/document_loader.py")
    vector_code = read_file("src/rag_engine/vector_store.py")
    generator_code = read_file("src/generator/llm_generator.py")
except Exception as e:
    print(f"Erreur de lecture: {e}")
    exit(1)

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🧠 Projet NLP & RAG - Intelligence Artificielle\n",
    "**Réalisé par : Amina & Firdawss** | **ENSA Al Hoceima - ID2**\n",
    "\n",
    "Bienvenue dans le Notebook officiel et autonome de notre projet. Ce document contient **l'intégralité du code source** de notre pipeline en deux phases principales :\n",
    "1. **Phase 1 : Boîte à outils NLP** (Pipelines de base avec HuggingFace)\n",
    "2. **Phase 2 : Chatbot RAG Intelligent** (Génération Augmentée par la Recherche sur nos propres cours PDF)\n",
    "\n",
    "Puisque ce Notebook est conçu pour tourner sur Google Colab, tout le code des différentes classes a été regroupé ici dans des cellules exécutables pour faciliter l'évaluation par le professeur.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🛠️ Étape 0 : Installation des Dépendances et Téléchargement des PDF\n",
    "Nous installons les bibliothèques nécessaires et téléchargeons les fichiers PDF originaux depuis notre dépôt GitHub."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q transformers torch sentence-transformers faiss-cpu PyPDF2\n",
    "!git clone https://github.com/amina-dourdi/nlp-transformers-rag.git\n",
    "import os\n",
    "os.chdir(\"nlp-transformers-rag\")\n",
    "print(\"✅ Environnement préparé et PDF téléchargés !\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## ⚙️ Configuration Globale (`config.py`)\n",
    "Définition des hyperparamètres et des modèles HuggingFace utilisés dans le projet."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": config_code.splitlines(True)
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 📚 Phase 1 : Boîte à outils NLP (`nlp_pipelines.py`)\n",
    "Fonctions de classification, résumé et Q&A classiques utilisant les `pipeline`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": nlp_code.splitlines(True)
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 🤖 Phase 2 : Ingestion des Documents RAG (`document_loader.py`)\n",
    "Lecture des PDF avec `PyPDF2` et découpage intelligent en 'Chunks' pour respecter la fenêtre de contexte."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": loader_code.splitlines(True)
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📊 Base de Données Vectorielle (`vector_store.py`)\n",
    "Encodage des Chunks avec `all-MiniLM-L6-v2` et indexation mathématique avec **FAISS**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": vector_code.splitlines(True)
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🧠 Générateur RAG Augmenté (`llm_generator.py`)\n",
    "Modèle `Bloomz-560m` qui génère la réponse finale en fusionnant la question avec les documents extraits par FAISS."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": generator_code.splitlines(True)
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# 🚀 Éxécution & Démonstration Complète\n",
    "Test du pipeline RAG avec une vraie question pour démontrer les capacités de l'Intelligence Artificielle sourcée."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"=== 1. INDEXATION DES PDF DE COURS ===\")\n",
    "docs = load_pdf_documents(\"data/corpus\")\n",
    "chunks_list = get_chunks(docs)\n",
    "manager = VectorStoreManager(EMBEDDING_MODEL_NAME, VECTOR_DB_PATH)\n",
    "manager.build_and_save_index(chunks_list)\n",
    "print(\"\\n=== 2. QUESTION UTILISATEUR ===\")\n",
    "q = \"Qu'est-ce que le mécanisme d'attention dans les Transformers ?\"\n",
    "print(f\"Question posée : '{q}'\")\n",
    "res = manager.search_top_k(q, k=3)\n",
    "print(\"\\n=== 3. GÉNÉRATION LLM (SANS RAG) ===\")\n",
    "gen = RAGGenerator()\n",
    "print(\"❌ \", gen.generate_without_rag(q))\n",
    "print(\"\\n=== 4. GÉNÉRATION LLM (AVEC RAG) ===\")\n",
    "print(\"✅ \", gen.generate(q, res))\n",
    "print(\"\\n=== 5. PREUVE DES SOURCES ===\")\n",
    "for i, r in enumerate(res, 1):\n",
    "    print(f\"Source {i} : {r['source']}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print("✅ Le fichier Final_Project_Colab.ipynb a été généré avec succès avec tout le code inclus !")
