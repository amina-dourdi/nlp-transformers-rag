# 📋 Guide Détaillé des Tâches — Firdawss & Amina

> Ce document explique **en détail** chaque tâche restante du projet.
> Pour chaque tâche, vous trouverez : le contexte, l'explication théorique, le code exact à écrire, et un exemple de test.

---

# ═══════════════════════════════════════════════════════
# 🟫 PARTIE 1 : TÂCHES DE FIRDAWSS
# ═══════════════════════════════════════════════════════

---

## 🟫 Tâche F1 : Implémenter la Classification de Texte

### 📁 Fichier à modifier : `src/core_nlp/nlp_pipelines.py`

### 🎯 Objectif
Ajouter une fonction `classify_text(text)` qui prend un texte en entrée et retourne
l'**émotion détectée** (joie, tristesse, colère, peur, surprise, amour) avec un score
de confiance.

### 📖 Contexte Théorique
La **classification de texte** est l'une des tâches fondamentales du NLP. Elle consiste
à assigner automatiquement une catégorie (ou label) à un texte donné.

Dans notre projet, on utilise le modèle `bhadresh-savani/distilbert-base-uncased-emotion`
qui est un modèle **DistilBERT** (version légère de BERT avec 6 couches au lieu de 12)
fine-tuné sur le dataset **Emotion** pour reconnaître 6 émotions :

| Label | Signification |
|---|---|
| `joy` | Joie |
| `sadness` | Tristesse |
| `anger` | Colère |
| `fear` | Peur |
| `surprise` | Surprise |
| `love` | Amour |

**Comment ça marche en coulisse :**
1. Le texte est tokenisé (découpé en tokens numériques).
2. Les tokens passent dans DistilBERT qui produit un vecteur contextuel.
3. Une couche de classification (tête linéaire) produit 6 scores (un par émotion).
4. Une fonction softmax convertit ces scores en probabilités (qui somment à 1).
5. L'émotion avec la plus haute probabilité est retournée.

```
"I am so happy today!" → Tokenisation → DistilBERT → [0.02, 0.95, 0.01, 0.01, 0.005, 0.005]
                                                        ↑sad  ↑joy  ↑ang  ↑fear ↑surp  ↑love
                                                  Résultat : joy (95%)
```

### 💻 Code à Écrire

**Étape 1 :** Ajouter une variable de cache globale (après la ligne `_qa_model = None`) :

```python
_classifier_pipeline = None
```

**Étape 2 :** Dans la fonction `init_pipelines()`, ajouter le chargement du classificateur :

```python
print("🔄 Initialisation du pipeline de Classification de Texte...")
global _classifier_pipeline
_classifier_pipeline = pipeline("text-classification", model=NLP_CLASSIFIER_MODEL)
```

**Étape 3 :** Ajouter la fonction de classification (après `answer_question`) :

```python
def classify_text(text: str) -> dict:
    """
    Classification de texte par émotion (Tâche Firdawss).
    Utilise le modèle DistilBERT fine-tuné sur le dataset Emotion.
    
    Référence : Livre "NLP with Transformers", Chapitres 2 & 9
    
    Args:
        text (str): Le texte à classifier.
    
    Returns:
        dict: {"label": str, "score": float}
              Exemple : {"label": "joy", "score": 0.98}
    """
    global _classifier_pipeline
    if _classifier_pipeline is None:
        _classifier_pipeline = pipeline("text-classification", model=NLP_CLASSIFIER_MODEL)
    
    # Le pipeline retourne une liste : [{"label": "joy", "score": 0.98}]
    result = _classifier_pipeline(text)
    return result[0]
```

### ✅ Comment Tester

```python
# Dans un notebook ou un script de test :
from src.core_nlp.nlp_pipelines import classify_text

# Test 1 : Joie
print(classify_text("I am so happy and excited about this project!"))
# Attendu : {"label": "joy", "score": ~0.97}

# Test 2 : Tristesse
print(classify_text("I feel so sad and lonely today."))
# Attendu : {"label": "sadness", "score": ~0.98}

# Test 3 : Colère
print(classify_text("This is absolutely unacceptable and infuriating!"))
# Attendu : {"label": "anger", "score": ~0.95}
```

---

## 🟫 Tâche F2 : Implémenter le Résumé Automatique

### 📁 Fichier à modifier : `src/core_nlp/nlp_pipelines.py`

### 🎯 Objectif
Ajouter une fonction `summarize_text(text, max_length, min_length)` qui prend un texte
long et retourne un **résumé condensé** de ce texte.

### 📖 Contexte Théorique
Le **résumé automatique** (Text Summarization) est une tâche qui consiste à produire une
version plus courte d'un texte tout en conservant les informations essentielles.

Il existe deux types de résumé :
- **Extractif** : Sélectionne les phrases les plus importantes du texte original.
- **Abstractif** : Génère de nouvelles phrases qui reformulent le contenu (c'est ce qu'on utilise ici).

Notre modèle est **BART** (`facebook/bart-large-cnn`) :
- BART = **B**idirectional and **A**uto-**R**egressive **T**ransformer.
- C'est un modèle **encodeur-décodeur** (contrairement à BERT qui est encodeur seul).
- Il a été fine-tuné sur le dataset **CNN/DailyMail** (articles de presse + résumés humains).
- L'encodeur comprend le texte source, et le décodeur génère le résumé mot par mot.

```
Texte long (500 mots) → Encodeur BART → Représentation interne → Décodeur BART → Résumé (50 mots)
```

**Paramètres importants :**
- `max_length` : Nombre maximum de tokens dans le résumé (défaut : 130).
- `min_length` : Nombre minimum de tokens dans le résumé (défaut : 30).
- Si le texte source est trop court, BART ne pourra pas bien résumer.

### 💻 Code à Écrire

**Étape 1 :** Ajouter une variable de cache globale :

```python
_summarizer_pipeline = None
```

**Étape 2 :** Dans `init_pipelines()`, ajouter :

```python
print("🔄 Initialisation du pipeline de Résumé Automatique...")
global _summarizer_pipeline
_summarizer_pipeline = pipeline("summarization", model=NLP_SUMMARIZER_MODEL)
```

**Étape 3 :** Ajouter la fonction de résumé :

```python
def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> dict:
    """
    Résumé automatique de texte (Tâche Firdawss).
    Utilise le modèle BART fine-tuné sur CNN/DailyMail.
    
    Référence : Livre "NLP with Transformers", Chapitre 6
    
    Args:
        text (str): Le texte long à résumer.
        max_length (int): Longueur maximale du résumé en tokens (défaut: 130).
        min_length (int): Longueur minimale du résumé en tokens (défaut: 30).
    
    Returns:
        dict: {"summary_text": str}
              Exemple : {"summary_text": "Le résumé du texte..."}
    """
    global _summarizer_pipeline
    if _summarizer_pipeline is None:
        _summarizer_pipeline = pipeline("summarization", model=NLP_SUMMARIZER_MODEL)
    
    # Le pipeline retourne : [{"summary_text": "Le résumé..."}]
    result = _summarizer_pipeline(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False  # Résumé déterministe (pas d'aléatoire)
    )
    return result[0]
```

### ⚠️ Attention
- Le texte en entrée doit être **assez long** (au moins 50 mots) sinon BART peut bugger.
- Le modèle `bart-large-cnn` pèse ~1.6 Go en RAM. Le premier appel prendra du temps.
- `do_sample=False` assure que le résumé est le même à chaque exécution (déterministe).

### ✅ Comment Tester

```python
from src.core_nlp.nlp_pipelines import summarize_text

texte_long = """
The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building,
and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on
each side. During its construction, the Eiffel Tower surpassed the Washington Monument to
become the tallest man-made structure in the world, a title it held for 41 years until the
Chrysler Building in New York City was finished in 1930. It was the first structure to reach
a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower
in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding
transmitters, the Eiffel Tower is the second tallest free-standing structure in France after
the Millau Viaduct.
"""

result = summarize_text(texte_long)
print(result)
# Attendu : {"summary_text": "The tower is 324 metres tall, about the same height as..."}
```

---

## 🟫 Tâche F3 : Mettre à jour le Cache Global dans `init_pipelines()`

### 📁 Fichier à modifier : `src/core_nlp/nlp_pipelines.py`

### 🎯 Objectif
Actuellement, `init_pipelines()` ne charge que les modèles d'Amina (sentiment + QA).
Il faut y ajouter le chargement des modèles de Firdawss pour que **tous les modèles
se chargent en une seule fois** au démarrage de l'application.

### 📖 Pourquoi c'est Important
Quand l'utilisateur lance Streamlit, on veut que **tous les modèles** se chargent en
mémoire dès le départ. Si on ne le fait pas, le premier clic sur "Classification" ou
"Résumé" sera très lent (5-30 secondes d'attente pendant le téléchargement du modèle).

### 💻 Code Final de `init_pipelines()`

La fonction complète doit ressembler à ça :

```python
_sentiment_pipeline = None
_qa_tokenizer = None
_qa_model = None
_classifier_pipeline = None    # ← NOUVEAU (Firdawss)
_summarizer_pipeline = None    # ← NOUVEAU (Firdawss)

def init_pipelines():
    """
    Initialise TOUS les modèles Hugging Face du projet.
    """
    global _sentiment_pipeline, _qa_tokenizer, _qa_model
    global _classifier_pipeline, _summarizer_pipeline  # ← NOUVEAU
    
    # ── Modèles d'Amina ──
    print("🔄 Initialisation du pipeline d'Analyse de Sentiment...")
    _sentiment_pipeline = pipeline("sentiment-analysis", model=NLP_SENTIMENT_MODEL)
    
    print("🔄 Initialisation des composants de Question Answering...")
    _qa_tokenizer = AutoTokenizer.from_pretrained(NLP_QA_MODEL)
    _qa_model = AutoModelForQuestionAnswering.from_pretrained(NLP_QA_MODEL)
    
    # ── Modèles de Firdawss ── (NOUVEAU)
    print("🔄 Initialisation du pipeline de Classification...")
    _classifier_pipeline = pipeline("text-classification", model=NLP_CLASSIFIER_MODEL)
    
    print("🔄 Initialisation du pipeline de Résumé Automatique...")
    _summarizer_pipeline = pipeline("summarization", model=NLP_SUMMARIZER_MODEL)
    
    print("✅ Tous les modèles NLP sont prêts !")
```

---

## 🟫 Tâche F4 : Implémenter le Chargement des PDF

### 📁 Fichier à modifier : `src/rag_engine/document_loader.py`

### 🎯 Objectif
Écrire le code qui **lit les fichiers PDF** stockés dans `data/corpus/` et extrait leur
contenu textuel page par page.

### 📖 Contexte Théorique
Un fichier PDF n'est pas du texte brut — c'est un format binaire complexe qui contient
des polices, des images, des mises en page, etc. Il faut utiliser une bibliothèque
spécialisée pour en extraire le texte.

On utilise **`pypdf`** (anciennement PyPDF2) qui est la bibliothèque standard Python
pour lire des PDF. Elle offre :
- `PdfReader(path)` : Ouvre un fichier PDF.
- `reader.pages` : Liste de toutes les pages du PDF.
- `page.extract_text()` : Extrait le texte d'une page.

**Schéma du flux :**
```
data/corpus/
  ├── cours_nlp.pdf        → PdfReader → Page 1 texte, Page 2 texte, ...
  └── cours_transformers.pdf → PdfReader → Page 1 texte, Page 2 texte, ...
                                             ↓
                                   Liste de dictionnaires :
                                   [{"text": "...", "source": "cours_nlp.pdf - Page 1"},
                                    {"text": "...", "source": "cours_nlp.pdf - Page 2"},
                                    ...]
```

### 💻 Code à Écrire

Remplacer le contenu de `document_loader.py` par :

```python
# src/rag_engine/document_loader.py
# ==========================================
# PHASE 2 : RAG - Chargement & Découpage
# ==========================================

# 🟫 Firdawss's Tasks:
# - Charger les documents PDF ou TXT
# - Découper les textes en chunks (segments)

from typing import List, Dict
import os
from pypdf import PdfReader
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf_documents(folder_path: str) -> List[Dict[str, str]]:
    """
    Charger tous les documents PDF du dossier corpus.
    
    Pour chaque PDF trouvé dans le dossier, on extrait le texte page par page.
    Chaque page devient un document séparé avec sa source (nom du fichier + numéro de page).
    
    Args:
        folder_path (str): Chemin vers le dossier contenant les PDF.
    
    Returns:
        List[Dict[str, str]]: Liste de dictionnaires avec les clés "text" et "source".
        Exemple : [
            {"text": "Contenu de la page 1...", "source": "cours_nlp.pdf - Page 1"},
            {"text": "Contenu de la page 2...", "source": "cours_nlp.pdf - Page 2"},
        ]
    """
    documents = []
    
    # Vérifier que le dossier existe
    if not os.path.exists(folder_path):
        print(f"⚠️ Le dossier '{folder_path}' n'existe pas !")
        return documents
    
    # Parcourir tous les fichiers du dossier
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            print(f"📄 Lecture de '{filename}'...")
            
            try:
                reader = PdfReader(filepath)
                
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    
                    # Ne garder que les pages qui contiennent du texte
                    if text and text.strip():
                        documents.append({
                            "text": text.strip(),
                            "source": f"{filename} - Page {page_num}"
                        })
                
                print(f"  ✅ {len(reader.pages)} pages extraites de '{filename}'")
                
            except Exception as e:
                print(f"  ❌ Erreur lors de la lecture de '{filename}': {e}")
    
    print(f"\n📚 Total : {len(documents)} pages extraites de {folder_path}")
    return documents


def get_chunks(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Découper les documents chargés en chunks (morceaux) de taille homogène.
    
    Chaque document (page) est découpé en segments de CHUNK_SIZE caractères
    avec un chevauchement de CHUNK_OVERLAP caractères entre segments consécutifs.
    Le chevauchement permet de ne pas couper une phrase ou une idée en plein milieu.
    
    Args:
        documents (List[Dict]): Liste de documents issus de load_pdf_documents().
    
    Returns:
        List[Dict[str, str]]: Liste de chunks avec les clés "text" et "source".
    """
    chunks = []
    
    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        
        # Si le texte est plus court que CHUNK_SIZE, le garder tel quel
        if len(text) <= CHUNK_SIZE:
            chunks.append({"text": text, "source": source})
            continue
        
        # Découper le texte en morceaux avec chevauchement
        start = 0
        chunk_num = 1
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]
            
            # Ne garder que les chunks non vides
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "source": f"{source} (Chunk {chunk_num})"
                })
                chunk_num += 1
            
            # Avancer la fenêtre (avec chevauchement)
            start += CHUNK_SIZE - CHUNK_OVERLAP
    
    print(f"✂️ {len(chunks)} chunks créés à partir de {len(documents)} documents.")
    return chunks
```

### 📖 Explication du Chunking avec Chevauchement

Pourquoi le **chevauchement** (overlap) est essentiel :

```
Sans chevauchement (MAUVAIS) :
──────────────────────────────────────────────
"Les Transformers ont été introduits | en 2017 par Google dans le célèbre"
         Chunk 1                     |         Chunk 2
         ↑ La phrase est coupée en deux ! Le sens est perdu.

Avec chevauchement de 50 caractères (BON) :
──────────────────────────────────────────────
"Les Transformers ont été introduits en 2017 par Google"
         Chunk 1 ──────────────────────────────┘
                           "introduits en 2017 par Google dans le célèbre article"
                            └────────────── Chunk 2
         ↑ Les 50 derniers caractères du chunk 1 sont répétés au début du chunk 2.
           L'idée n'est jamais coupée !
```

### ✅ Comment Tester

```python
from src.rag_engine.document_loader import load_pdf_documents, get_chunks
from src.config import CORPUS_DIR

# 1. Charger les PDF
documents = load_pdf_documents(CORPUS_DIR)
print(f"Nombre de pages : {len(documents)}")
print(f"Première page : {documents[0]['source']}")
print(f"Aperçu : {documents[0]['text'][:200]}...")

# 2. Découper en chunks
chunks = get_chunks(documents)
print(f"\nNombre de chunks : {len(chunks)}")
print(f"Premier chunk : {chunks[0]['source']}")
print(f"Taille du chunk : {len(chunks[0]['text'])} caractères")
```

---

## 🟫 Tâche F5 : Ajouter la Persistance de l'Index FAISS

### 📁 Fichier à modifier : `src/rag_engine/vector_store.py`

### 🎯 Objectif
Actuellement, l'index FAISS est créé en mémoire mais **jamais sauvegardé sur le disque**.
Cela signifie qu'à chaque redémarrage, il faut recalculer tous les embeddings (très lent !).
Il faut ajouter les fonctions `save_index()` et `load_index()`.

### 📖 Contexte Théorique
FAISS fournit deux fonctions natives pour la persistance :
- `faiss.write_index(index, path)` : Sauvegarde l'index dans un fichier binaire.
- `faiss.read_index(path)` : Recharge l'index depuis le fichier.

Mais l'index FAISS ne stocke que les **vecteurs numériques**, pas les textes originaux.
Il faut donc aussi sauvegarder la liste `self.documents` séparément (avec `json` ou `pickle`).

### 💻 Code à Ajouter

Ajouter ces deux méthodes dans la classe `VectorStoreManager` :

```python
import json  # Ajouter en haut du fichier avec les autres imports

# Dans la classe VectorStoreManager :

def save_index(self):
    """
    Sauvegarder l'index FAISS et les métadonnées des documents sur le disque.
    """
    if self.index is None:
        print("⚠️ Aucun index à sauvegarder !")
        return
    
    # Créer le dossier s'il n'existe pas
    os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
    
    # 1. Sauvegarder l'index FAISS
    faiss.write_index(self.index, self.index_path)
    print(f"💾 Index FAISS sauvegardé dans '{self.index_path}'")
    
    # 2. Sauvegarder les documents associés (textes + sources)
    docs_path = self.index_path + ".docs.json"
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(self.documents, f, ensure_ascii=False, indent=2)
    print(f"💾 Métadonnées sauvegardées dans '{docs_path}'")

def load_index(self):
    """
    Charger un index FAISS précédemment sauvegardé depuis le disque.
    """
    if not os.path.exists(self.index_path):
        print(f"⚠️ Aucun index trouvé dans '{self.index_path}'")
        return False
    
    # 1. Charger l'index FAISS
    self.index = faiss.read_index(self.index_path)
    print(f"📂 Index FAISS chargé ({self.index.ntotal} vecteurs)")
    
    # 2. Charger les documents associés
    docs_path = self.index_path + ".docs.json"
    if os.path.exists(docs_path):
        with open(docs_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        print(f"📂 {len(self.documents)} documents chargés")
    
    return True
```

Et modifier `build_and_save_index()` pour **sauvegarder automatiquement** :

```python
def build_and_save_index(self, chunks: List[Dict[str, str]]):
    # ... (code existant inchangé) ...
    print(f"✅ Index FAISS créé avec {self.index.ntotal} documents.")
    
    # NOUVEAU : Sauvegarder sur disque
    self.save_index()
```

### ✅ Comment Tester

```python
from src.rag_engine.vector_store import VectorStoreManager
from src.config import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH

# Créer et sauvegarder
manager = VectorStoreManager(EMBEDDING_MODEL_NAME, VECTOR_DB_PATH)
fake_chunks = [
    {"text": "Les Transformers sont une architecture de deep learning.", "source": "test.pdf"},
    {"text": "FAISS est une bibliothèque de recherche vectorielle.", "source": "test.pdf"},
]
manager.build_and_save_index(fake_chunks)

# Recharger depuis le disque (simule un redémarrage)
manager2 = VectorStoreManager(EMBEDDING_MODEL_NAME, VECTOR_DB_PATH)
manager2.load_index()
results = manager2.search_top_k("deep learning", k=1)
print(results)  # Doit retourner le chunk sur les Transformers
```

---

## 🟫 Tâche F6 : Compléter le Notebook Phase 1

### 📁 Fichier à modifier : `notebooks/01_exploration_transformers.ipynb`

### 🎯 Objectif
Créer un notebook Jupyter qui **importe et teste les 4 pipelines NLP** (Sentiment,
QA, Classification, Résumé) avec des exemples clairs et des résultats affichés.

### 💻 Cellules à Écrire

**Cellule 1 : Imports**
```python
import sys
sys.path.insert(0, "..")  # Pour pouvoir importer src/

from src.core_nlp.nlp_pipelines import (
    analyze_sentiment,
    answer_question,
    classify_text,
    summarize_text
)
```

**Cellule 2 : Test Analyse de Sentiment (Amina)**
```python
print("=" * 50)
print("🟦 TEST 1 : Analyse de Sentiment (Amina)")
print("=" * 50)

textes_test = [
    "I absolutely love this NLP course, it's amazing!",
    "This homework is terrible and boring.",
    "The weather is okay today, nothing special."
]

for texte in textes_test:
    result = analyze_sentiment(texte)
    print(f"\nTexte : \"{texte}\"")
    print(f"  → Sentiment : {result['label']} (confiance : {result['score']:.2%})")
```

**Cellule 3 : Test Question Answering (Amina)**
```python
print("=" * 50)
print("🟦 TEST 2 : Question Answering (Amina)")
print("=" * 50)

context = """
Le Natural Language Processing (NLP) est une branche de l'intelligence artificielle
qui s'intéresse à la compréhension et au traitement du langage humain par les machines.
Les Transformers, introduits en 2017 par Google dans l'article "Attention Is All You Need",
ont révolutionné le domaine en permettant le traitement parallèle des séquences textuelles.
"""

questions = [
    "Qu'est-ce que le NLP ?",
    "Quand les Transformers ont-ils été introduits ?",
    "Qui a introduit les Transformers ?"
]

for q in questions:
    result = answer_question(q, context)
    print(f"\nQuestion : \"{q}\"")
    print(f"  → Réponse : \"{result['answer']}\" (score : {result['score']:.2f})")
```

**Cellule 4 : Test Classification (Firdawss)**
```python
print("=" * 50)
print("🟫 TEST 3 : Classification de Texte (Firdawss)")
print("=" * 50)

textes_emotions = [
    "I'm thrilled about graduating this year!",
    "I'm terrified of the final exam.",
    "My best friend betrayed me, I'm furious.",
    "I miss my family so much...",
]

for texte in textes_emotions:
    result = classify_text(texte)
    print(f"\nTexte : \"{texte}\"")
    print(f"  → Émotion : {result['label']} (confiance : {result['score']:.2%})")
```

**Cellule 5 : Test Résumé (Firdawss)**
```python
print("=" * 50)
print("🟫 TEST 4 : Résumé Automatique (Firdawss)")
print("=" * 50)

texte_long = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
Locally nicknamed "La dame de fer", it was constructed from 1887 to 1889 as the centerpiece
of the 1889 World's Fair. Although initially criticised by some of France's leading artists
and intellectuals for its design, it has since become a global cultural icon of France and
one of the most recognizable structures in the world. The tower is 330 metres tall and the
tallest structure in Paris. Its base is square, measuring 125 metres on each side. The tower
has three levels for visitors, with restaurants on the first and second levels.
"""

result = summarize_text(texte_long)
print(f"Texte original ({len(texte_long)} caractères) :")
print(f"  {texte_long[:150]}...")
print(f"\nRésumé :")
print(f"  {result['summary_text']}")
```

---

# ═══════════════════════════════════════════════════════
# 🟦 PARTIE 2 : TÂCHES D'AMINA
# ═══════════════════════════════════════════════════════

---

## 🟦 Tâche A1 : Connecter un Vrai LLM dans le Générateur

### 📁 Fichier à modifier : `src/generator/llm_generator.py`

### 🎯 Objectif
Remplacer le `return` statique dans `generate()` par un véritable appel à un modèle
de langage génératif. Le LLM recevra le prompt augmenté (question + contexte RAG) et
produira une réponse en langage naturel.

### 📖 Contexte Théorique
Un **LLM** (Large Language Model) est un modèle de langage entraîné sur d'énormes
quantités de texte. Il est capable de **générer du texte** en prédisant le prochain
mot (token) à partir du contexte précédent.

Dans notre projet, le LLM reçoit un **prompt augmenté** contenant :
1. Les instructions (rôle de l'assistant).
2. Le contexte extrait des documents (les chunks trouvés par FAISS).
3. La question de l'utilisateur.

Le LLM génère ensuite une réponse basée **uniquement** sur ce contexte.

**3 options possibles :**

| Option | Avantage | Inconvénient |
|---|---|---|
| **A. API HuggingFace Inference** | Gratuit, simple, pas besoin de GPU | Limité en nombre de requêtes |
| **B. API Google Gemini** | Très puissant, gratuit (tier limité) | Nécessite une clé API Google |
| **C. Modèle local (Gemma 2B)** | Fonctionne hors ligne | Nécessite ~8 Go de RAM et est lent sans GPU |

### 💻 Option A : Avec l'API HuggingFace Inference (Recommandée)

```python
# src/generator/llm_generator.py
from typing import List, Dict
from src.config import GENERATIVE_LLM_MODEL

# Installer si nécessaire : pip install huggingface_hub
from huggingface_hub import InferenceClient


class RAGGenerator:
    def __init__(self, model_name_or_api_key: str = GENERATIVE_LLM_MODEL):
        """
        Initialise le client d'inférence HuggingFace.
        
        Pour obtenir un token gratuit :
        1. Créer un compte sur https://huggingface.co
        2. Aller dans Settings → Access Tokens → New Token
        3. Copier le token (commence par "hf_...")
        """
        self.client = InferenceClient(
            model=model_name_or_api_key,
            token="hf_VOTRE_TOKEN_ICI"  # ← Remplacer par votre token
        )
        print(f"✅ Générateur LLM connecté au modèle '{model_name_or_api_key}'")

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
        Assemble le prompt et génère la réponse via l'API HuggingFace.
        """
        prompt = self.build_prompt(query, contexts)
        
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=512,     # Longueur max de la réponse
                temperature=0.3,        # Peu d'aléatoire (réponses précises)
                repetition_penalty=1.2  # Éviter les répétitions
            )
            return response
        except Exception as e:
            return f"❌ Erreur de génération : {e}"
    
    def generate_without_rag(self, query: str) -> str:
        """
        Génère une réponse SANS contexte RAG (pour la comparaison).
        """
        prompt = f"""Répondez à la question suivante de manière claire et détaillée :

Question : {query}

Réponse :"""
        
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=512,
                temperature=0.3
            )
            return response
        except Exception as e:
            return f"❌ Erreur de génération : {e}"
```

### 💻 Option B : Avec l'API Google Gemini

```python
# Si vous préférez Gemini, modifier le __init__ et generate() :
# pip install google-generativeai

import google.generativeai as genai

class RAGGenerator:
    def __init__(self, model_name_or_api_key: str = "votre_clé_api"):
        genai.configure(api_key=model_name_or_api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
    
    def generate(self, query, contexts):
        prompt = self.build_prompt(query, contexts)
        response = self.model.generate_content(prompt)
        return response.text
```

### 💻 Option C : Avec un Pipeline Local (Transformers)

```python
# Si vous avez assez de RAM (~8 Go) :
from transformers import pipeline

class RAGGenerator:
    def __init__(self, model_name_or_api_key: str = GENERATIVE_LLM_MODEL):
        self.generator = pipeline(
            "text-generation",
            model=model_name_or_api_key,
            device_map="auto"  # Utilise GPU si disponible
        )
    
    def generate(self, query, contexts):
        prompt = self.build_prompt(query, contexts)
        result = self.generator(prompt, max_new_tokens=512, do_sample=True, temperature=0.3)
        # Extraire uniquement la partie générée (pas le prompt)
        generated = result[0]["generated_text"][len(prompt):]
        return generated.strip()
```

### ⚠️ Important
- **Ne commite jamais ta clé API dans Git !** Utilise plutôt une variable d'environnement :
  ```python
  import os
  token = os.environ.get("HF_TOKEN", "")
  ```
- La fonction `generate_without_rag()` est **essentielle** pour la démo comparative.

---

## 🟦 Tâche A2 : Ajouter des PDF de Cours dans le Corpus

### 📁 Dossier cible : `data/corpus/`

### 🎯 Objectif
Le dossier `data/corpus/` est actuellement **vide** (il ne contient qu'un README.md).
Il faut y placer au minimum **2-3 fichiers PDF de cours** qui serviront de base de
connaissances pour le système RAG.

### 📖 Ce qu'il Faut Faire
1. Choisir 2-3 PDF de vos cours (NLP, IA, Data Engineering, etc.).
2. Les copier dans `data/corpus/`.
3. S'assurer que les PDF contiennent du **texte sélectionnable** (pas des scans d'images).

### ⚠️ Attention
- Les PDF scannés (images) ne fonctionneront **PAS** avec `pypdf` — il faut des PDF
  avec du texte réel.
- Le fichier `Natural language processing with Transformers new version.pdf` qui est
  dans le dossier parent (`projet/`) peut être copié dans `data/corpus/`.
- Testez que `pypdf` peut bien extraire le texte :
  ```python
  from pypdf import PdfReader
  reader = PdfReader("data/corpus/votre_fichier.pdf")
  print(reader.pages[0].extract_text()[:500])
  ```

---

## 🟦 Tâche A3 : Développer l'Onglet 1 de Streamlit (NLP)

### 📁 Fichier à modifier : `app.py`

### 🎯 Objectif
Transformer l'onglet "Boîte à outils NLP" qui est actuellement vide en une interface
interactive permettant de tester les 4 pipelines NLP.

### 💻 Code à Écrire dans le `with tab1:`

```python
with tab1:
    st.header("Phase 1 : Exploration NLP")
    st.write("Testez les tâches NLP implémentées par Amina et Firdawss.")
    
    # Importer les fonctions NLP
    from src.core_nlp.nlp_pipelines import (
        analyze_sentiment, answer_question, classify_text, summarize_text
    )
    
    # ── Sous-onglets pour chaque tâche ──
    nlp_tab1, nlp_tab2, nlp_tab3, nlp_tab4 = st.tabs([
        "😊 Sentiment", "🏷️ Classification", "❓ Question Answering", "📝 Résumé"
    ])
    
    # ── 1. Analyse de Sentiment (Amina) ──
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
    
    # ── 2. Classification (Firdawss) ──
    with nlp_tab2:
        st.subheader("Classification de Texte (Émotions)")
        st.write("Détectez l'émotion dominante d'un texte (joie, tristesse, colère, peur, surprise, amour).")
        
        classif_input = st.text_area(
            "Entrez votre texte :", 
            value="I am so excited about this project!",
            key="classif_input"
        )
        
        if st.button("🏷️ Classifier", key="btn_classif"):
            with st.spinner("Classification en cours..."):
                result = classify_text(classif_input)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Émotion", result["label"])
            with col2:
                st.metric("Confiance", f"{result['score']:.2%}")
    
    # ── 3. Question Answering (Amina) ──
    with nlp_tab3:
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
    
    # ── 4. Résumé Automatique (Firdawss) ──
    with nlp_tab4:
        st.subheader("Résumé Automatique")
        st.write("Entrez un texte long et obtenez un résumé condensé.")
        
        summary_input = st.text_area(
            "Texte à résumer :",
            value="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris...",
            height=200,
            key="summary_input"
        )
        
        col_min, col_max = st.columns(2)
        with col_min:
            min_len = st.slider("Longueur min", 10, 100, 30, key="min_len")
        with col_max:
            max_len = st.slider("Longueur max", 50, 300, 130, key="max_len")
        
        if st.button("📝 Résumer", key="btn_summary"):
            with st.spinner("Résumé en cours..."):
                result = summarize_text(summary_input, max_length=max_len, min_length=min_len)
            
            st.success("**Résumé :**")
            st.write(result["summary_text"])
```

---

## 🟦 Tâche A4 : Développer l'Onglet 2 de Streamlit (RAG)

### 📁 Fichier à modifier : `app.py`

### 🎯 Objectif
Créer l'interface du chatbot RAG qui permet de :
1. Charger et indexer les documents PDF.
2. Poser des questions et recevoir des réponses basées sur les documents.
3. Comparer les réponses avec et sans RAG.

### 💻 Code à Écrire dans le `with tab2:`

```python
with tab2:
    st.header("Phase 2 : Système RAG")
    st.write("Posez des questions sur vos cours. Le système recherche dans vos documents pour formuler une réponse précise.")
    
    from src.rag_engine.document_loader import load_pdf_documents, get_chunks
    from src.rag_engine.vector_store import VectorStoreManager
    from src.generator.llm_generator import RAGGenerator
    from src.config import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH, CORPUS_DIR
    
    # ── Bouton d'indexation ──
    st.subheader("📂 1. Indexation des Documents")
    st.write(f"Documents source : `{CORPUS_DIR}`")
    
    if st.button("🔄 Indexer les Documents", key="btn_index"):
        with st.spinner("Chargement des PDF et création de l'index FAISS..."):
            documents = load_pdf_documents(CORPUS_DIR)
            chunks = get_chunks(documents)
            
            manager = VectorStoreManager(EMBEDDING_MODEL_NAME, VECTOR_DB_PATH)
            manager.build_and_save_index(chunks)
            
            st.session_state["vector_store"] = manager
        st.success(f"✅ {len(chunks)} chunks indexés avec succès !")
    
    st.divider()
    
    # ── Interface de Question ──
    st.subheader("💬 2. Posez votre Question")
    
    user_question = st.text_input(
        "Votre question :",
        placeholder="Ex: Qu'est-ce que le mécanisme d'attention ?",
        key="rag_question"
    )
    
    col_rag, col_no_rag = st.columns(2)
    
    if st.button("🚀 Obtenir la Réponse", key="btn_rag"):
        if "vector_store" not in st.session_state:
            st.warning("⚠️ Veuillez d'abord indexer les documents !")
        elif not user_question:
            st.warning("⚠️ Veuillez entrer une question !")
        else:
            manager = st.session_state["vector_store"]
            generator = RAGGenerator()
            
            # Recherche des documents pertinents
            with st.spinner("🔍 Recherche dans les documents..."):
                top_chunks = manager.search_top_k(user_question, k=3)
            
            # Génération AVEC RAG
            with col_rag:
                st.subheader("✅ Avec RAG")
                with st.spinner("Génération de la réponse..."):
                    response_rag = generator.generate(user_question, top_chunks)
                st.write(response_rag)
                
                st.divider()
                st.write("**📎 Sources utilisées :**")
                for i, chunk in enumerate(top_chunks, 1):
                    with st.expander(f"Source {i} : {chunk['source']}"):
                        st.write(chunk["text"][:300] + "...")
            
            # Génération SANS RAG (comparaison)
            with col_no_rag:
                st.subheader("❌ Sans RAG")
                with st.spinner("Génération sans contexte..."):
                    response_no_rag = generator.generate_without_rag(user_question)
                st.write(response_no_rag)
```

---

## 🟦 Tâche A5 : Compléter le Notebook Phase 2 (RAG)

### 📁 Fichier à modifier : `notebooks/02_experimentation_rag.ipynb`

### 🎯 Objectif
Tester le pipeline RAG complet dans un notebook pour valider chaque étape.

### 💻 Cellules à Écrire

```python
# Cellule 1 : Imports
import sys
sys.path.insert(0, "..")

from src.rag_engine.document_loader import load_pdf_documents, get_chunks
from src.rag_engine.vector_store import VectorStoreManager
from src.generator.llm_generator import RAGGenerator
from src.config import *

# Cellule 2 : Charger les PDF
documents = load_pdf_documents(CORPUS_DIR)
print(f"📄 {len(documents)} pages chargées")

# Cellule 3 : Découper en chunks
chunks = get_chunks(documents)
print(f"✂️ {len(chunks)} chunks créés")

# Cellule 4 : Créer l'index FAISS
manager = VectorStoreManager(EMBEDDING_MODEL_NAME, VECTOR_DB_PATH)
manager.build_and_save_index(chunks)

# Cellule 5 : Tester la recherche
query = "Qu'est-ce que le mécanisme d'attention ?"
results = manager.search_top_k(query, k=3)
for i, r in enumerate(results, 1):
    print(f"\n--- Résultat {i} ({r['source']}) ---")
    print(r["text"][:200])

# Cellule 6 : Générer une réponse RAG
generator = RAGGenerator()
response = generator.generate(query, results)
print("🤖 Réponse RAG :", response)

# Cellule 7 : Comparer avec/sans RAG
response_no_rag = generator.generate_without_rag(query)
print("❌ Sans RAG :", response_no_rag)
print("\n✅ Avec RAG :", response)
```

---

## 🟦 Tâche A6 : Tests Comparatifs avec/sans RAG

### 🎯 Objectif
Préparer **5 questions spécifiques** tirées des cours et montrer que :
- **Sans RAG** : le LLM hallucine ou donne des réponses vagues/incorrectes.
- **Avec RAG** : le LLM donne des réponses précises basées sur les documents.

### 💻 Questions à Préparer

```python
questions_test = [
    "Qu'est-ce que le mécanisme d'attention dans les Transformers ?",
    "Quelle est la différence entre BERT et GPT ?",
    "Comment fonctionne la tokenisation WordPiece ?",
    "Qu'est-ce que le fine-tuning d'un modèle pré-entraîné ?",
    "Expliquez le concept de transfer learning en NLP."
]

# Pour chaque question, afficher côte à côte :
# - La réponse SANS RAG (potentiellement fausse ou vague)
# - La réponse AVEC RAG (précise, avec sources)
```

### 📸 Captures d'Écran
Prendre des captures d'écran des résultats dans Streamlit pour le rapport final.

---

## 🟦 Tâche A7 : Rédiger le Rapport PDF Final

### 🎯 Objectif
Rédiger un rapport académique complet (co-rédigé avec Firdawss) contenant :

### 📝 Structure du Rapport

```
1. Page de Garde
   - Titre du projet
   - Noms : Amina & Firdawss
   - ENSA Al Hoceima - ID2 - S2

2. Introduction
   - Contexte : le NLP et ses applications
   - Objectifs du projet

3. État de l'Art
   - Les Transformers (historique, architecture)
   - Fine-tuning vs Prompt Engineering vs RAG
   - Pourquoi le RAG résout le problème des hallucinations

4. Architecture du Projet
   - Diagramme Mermaid (copier celui du plan)
   - Explication de chaque module
   - Technologies utilisées

5. Phase 1 : Pipelines NLP
   - Explication des 4 tâches
   - Code clé + résultats
   - Captures d'écran des tests

6. Phase 2 : Système RAG
   - Pipeline d'ingestion (PDF → chunks → embeddings → FAISS)
   - Prompt engineering et génération LLM
   - Résultats comparatifs (avec vs sans RAG)
   - Captures d'écran de l'interface Streamlit

7. Conclusion
   - Résultats obtenus
   - Limites du projet
   - Perspectives d'amélioration

8. Références
   - Livre "NLP with Transformers"
   - Documentation Hugging Face
   - Article "Attention Is All You Need"
```

---

# ═══════════════════════════════════════════════════════
# 📊 RÉCAPITULATIF FINAL
# ═══════════════════════════════════════════════════════

## Résumé Visuel

```
🟫 FIRDAWSS (7 tâches)                    🟦 AMINA (7 tâches)
────────────────────────                   ────────────────────────
F1 ✏️ classify_text()                     A1 ✏️ Connecter le LLM
F2 ✏️ summarize_text()                    A2 ✏️ Ajouter les PDF au corpus
F3 ✏️ Mettre à jour le cache             A3 ✏️ Onglet 1 Streamlit (NLP)
F4 ✏️ load_pdf_documents()               A4 ✏️ Onglet 2 Streamlit (RAG)
F5 ✏️ Persistance FAISS                  A5 ✏️ Notebook Phase 2
F6 ✏️ Notebook Phase 1                   A6 ✏️ Tests comparatifs
                                          A7 ✏️ Rapport PDF (avec Firdawss)
```

> **Dernière mise à jour** : 17 Juin 2026
