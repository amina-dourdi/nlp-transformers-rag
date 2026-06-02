# 📚 Explication Complète du Projet : NLP Transformers & Systèmes RAG

> **Projet Académique** — Mini-Projet NLP, ENSA Al Hoceima (Ingénierie des Données — S2)
> **Réalisé par** : Amina & Firdawss
> **Livre de Référence** : *Natural Language Processing with Transformers: Building Language Applications with Hugging Face*
> **Date** : Juin 2026

---

## Table des Matières

1. [Introduction Générale](#1--introduction-générale)
2. [Concepts Théoriques Fondamentaux](#2--concepts-théoriques-fondamentaux)
3. [Architecture Technique du Projet](#3--architecture-technique-du-projet)
4. [Dépendances & Installation](#4--dépendances--installation)
5. [Phase 1 — Pipelines NLP avec Transformers](#5--phase-1--pipelines-nlp-avec-transformers)
6. [Phase 2 — Système RAG (Retrieval-Augmented Generation)](#6--phase-2--système-rag-retrieval-augmented-generation)
7. [Interface Utilisateur Streamlit](#7--interface-utilisateur-streamlit)
8. [Répartition des Tâches entre Amina & Firdawss](#8--répartition-des-tâches-entre-amina--firdawss)
9. [État d'Avancement Actuel](#9--état-davancement-actuel)
10. [Prochaines Étapes & Travail Restant](#10--prochaines-étapes--travail-restant)

---

## 1. 🌍 Introduction Générale

### 1.1 Qu'est-ce que le NLP (Natural Language Processing) ?

Le **Traitement Automatique du Langage Naturel** (NLP) est une branche de l'intelligence artificielle
qui s'occupe de l'interaction entre les ordinateurs et le langage humain. L'objectif du NLP est de
permettre aux machines de **lire, comprendre, interpréter et générer** du texte de manière utile.

Des exemples concrets de NLP dans la vie quotidienne :
- **Google Translate** : Traduction automatique d'une langue à une autre.
- **Siri / Alexa** : Assistants vocaux qui comprennent vos commandes.
- **ChatGPT** : Génération de texte et conversation en langage naturel.
- **Filtres anti-spam** : Classification automatique des e-mails.

### 1.2 Objectif du Projet

Ce mini-projet a pour but de :

1. **Explorer les modèles Transformers pré-entraînés** de Hugging Face pour accomplir 4 tâches
   classiques du NLP (Classification, Sentiment, QA, Résumé).
2. **Construire un système RAG complet** (Retrieval-Augmented Generation) capable de répondre
   à des questions sur un corpus de documents académiques (PDF de cours) en combinant la
   recherche sémantique vectorielle avec un modèle de langage génératif.
3. **Développer une interface Web interactive** sous Streamlit pour démontrer le tout.

### 1.3 Pourquoi ce Projet est Important ?

Ce projet couvre l'ensemble du pipeline NLP moderne :

```
Données brutes (PDF) → Preprocessing → Modèle IA → Post-processing → Résultat utile
```

Vous apprendrez non seulement à **utiliser** des modèles IA, mais aussi à comprendre **comment
ils fonctionnent sous le capot** (tokenisation, tenseurs, logits, embeddings, etc.).

---

## 2. 🧠 Concepts Théoriques Fondamentaux

### 2.1 Les Transformers — La Révolution de 2017

Les **Transformers** sont une architecture de réseau de neurones introduite par Google dans le
célèbre article *"Attention Is All You Need"* (2017). Avant les Transformers, on utilisait des
réseaux récurrents (RNN, LSTM) qui traitaient le texte mot par mot, de façon séquentielle.

**Le problème des RNN** : ils "oublient" les mots du début quand la phrase est longue.

**La solution des Transformers** : le mécanisme d'**Attention** permet au modèle de regarder
**tous les mots en même temps** et de comprendre les relations entre eux, peu importe leur
position dans la phrase.

#### Schéma simplifié d'un Transformer :

```
Entrée texte → Tokenisation → Embeddings → [Encodeur Transformer] → Sortie contextuelle
                                                   ↕
                                          Mécanisme d'Attention
                                     (chaque mot regarde tous les autres)
```

### 2.2 Hugging Face — L'Écosystème

**Hugging Face** est une plateforme open-source qui héberge des milliers de modèles pré-entraînés.
Au lieu d'entraîner un modèle depuis zéro (ce qui coûte des millions de dollars et des semaines
de calcul), vous pouvez télécharger un modèle déjà entraîné et l'utiliser immédiatement.

Les outils clés de Hugging Face utilisés dans ce projet :
- **`transformers`** : Bibliothèque Python pour charger et utiliser des modèles.
- **`AutoTokenizer`** : Charge automatiquement le bon tokenizer pour un modèle donné.
- **`AutoModel`** : Charge automatiquement le bon modèle pré-entraîné.
- **`pipeline()`** : Fonction magique qui encapsule tokenisation + modèle + post-processing.

### 2.3 Le Tokenizer — Convertir du Texte en Nombres

Les modèles d'IA ne comprennent pas les mots. Ils ne comprennent que les **nombres**. Le
**Tokenizer** est l'outil qui convertit un texte humain en une séquence de nombres (appelés
**tokens**).

Exemple :
```
Texte :      "J'adore le NLP"
Tokens :     ["J'", "adore", "le", "NL", "##P"]
Token IDs :  [1001, 5234, 102, 8923, 4521]
```

Chaque modèle a son propre tokenizer. C'est pourquoi on utilise `AutoTokenizer.from_pretrained()`
qui télécharge automatiquement le bon tokenizer associé au modèle choisi.

### 2.4 Le Pipeline Hugging Face — Le Mode Facile

La fonction `pipeline()` est le moyen le plus simple d'utiliser un modèle. Elle cache toute
la complexité :

```python
from transformers import pipeline

# En UNE seule ligne, on fait de l'analyse de sentiment :
classificateur = pipeline("sentiment-analysis")
resultat = classificateur("I love this project!")
# Résultat : [{'label': 'POSITIVE', 'score': 0.9998}]
```

**Ce que `pipeline()` fait en coulisse :**
1. **Pré-processing** : Tokenise le texte → convertit en tenseurs PyTorch.
2. **Inférence** : Passe les tenseurs dans le modèle → obtient des logits (scores bruts).
3. **Post-processing** : Convertit les logits en labels lisibles (POSITIVE/NEGATIVE) + scores.

### 2.5 Les Embeddings — Représenter le Sens

Un **embedding** est un vecteur numérique (une liste de nombres) qui capture le **sens
sémantique** d'un texte. Deux textes qui parlent du même sujet auront des embeddings proches
dans l'espace vectoriel.

```
"Le chat dort sur le canapé"  →  [0.12, 0.85, -0.34, 0.67, ...]  (384 dimensions)
"Le félin sommeille sur le sofa" → [0.11, 0.83, -0.31, 0.69, ...]  (très proche !)
"Python est un langage"       →  [0.91, -0.22, 0.45, -0.18, ...]  (très différent)
```

Dans ce projet, on utilise le modèle `all-MiniLM-L6-v2` de `sentence-transformers` qui
transforme n'importe quel texte en un vecteur de **384 dimensions**.

### 2.6 FAISS — La Recherche Vectorielle Ultra-Rapide

**FAISS** (Facebook AI Similarity Search) est une bibliothèque de Meta/Facebook qui permet
de stocker des millions de vecteurs et de trouver les plus proches d'un vecteur donné en
quelques millisecondes.

**Comment ça marche :**
1. On stocke tous les embeddings de nos documents dans un index FAISS.
2. Quand l'utilisateur pose une question, on calcule l'embedding de sa question.
3. FAISS trouve les K documents dont les embeddings sont les plus proches → ce sont les
   documents les plus pertinents pour répondre à la question.

### 2.7 RAG — Retrieval-Augmented Generation

Le **RAG** est une technique qui améliore les réponses d'un modèle de langage (LLM) en lui
fournissant des **documents pertinents** comme contexte supplémentaire.

**Pourquoi le RAG est nécessaire ?**
- Les LLM sont entraînés sur des données générales et anciennes.
- Ils peuvent **halluciner** (inventer des informations fausses mais crédibles).
- Ils ne connaissent pas vos documents privés (vos cours, vos PDF).

**Le pipeline RAG en 3 étapes :**

```
┌──────────────────────────────────────────────────────────────────┐
│                     PIPELINE RAG COMPLET                         │
│                                                                  │
│  1. INDEXATION (une seule fois)                                  │
│     PDF → Extraction texte → Découpage en chunks → Embeddings    │
│     → Stockage dans FAISS                                        │
│                                                                  │
│  2. RETRIEVAL (à chaque question)                                │
│     Question utilisateur → Embedding de la question              │
│     → Recherche FAISS → Top-K chunks pertinents                  │
│                                                                  │
│  3. GENERATION                                                   │
│     Prompt = Question + Chunks pertinents                        │
│     → Envoi au LLM → Réponse précise basée sur vos documents     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 🏗️ Architecture Technique du Projet

### 3.1 Vue d'Ensemble de l'Arborescence

Le projet suit une architecture **modulaire** qui sépare les responsabilités en modules
indépendants. Cela permet à Amina et Firdawss de travailler en parallèle sans conflits.

```
nlp-transformers-rag/
│
├── data/                          # Données du projet
│   └── corpus/                    # Dossier pour stocker les PDF de cours
│
├── notebooks/                     # Notebooks Jupyter pour prototypage
│   ├── 01_exploration_transformers.ipynb   # Tests Phase 1
│   └── 02_experimentation_rag.ipynb        # Tests Phase 2
│
├── src/                           # Code source principal
│   ├── __init__.py                # Marque 'src' comme un package Python
│   ├── config.py                  # Configuration centralisée
│   │
│   ├── core_nlp/                  # ── Module NLP (Phase 1) ──
│   │   ├── __init__.py
│   │   └── nlp_pipelines.py       # Les 4 tâches NLP
│   │
│   ├── rag_engine/                # ── Module RAG : Ingestion & Recherche ──
│   │   ├── __init__.py
│   │   ├── document_loader.py     # Chargement PDF + découpage en chunks
│   │   ├── embedding_model.py     # Calcul des embeddings
│   │   └── vector_store.py        # Gestion de l'index FAISS
│   │
│   └── generator/                 # ── Module RAG : Génération ──
│       ├── __init__.py
│       └── llm_generator.py       # Prompt engineering + appel LLM
│
├── app.py                         # Interface Web Streamlit
├── requirements.txt               # Dépendances Python
├── README.md                      # Documentation
└── plan_projet_transformers_rag.md # Plan détaillé du projet
```

### 3.2 Pourquoi cette Architecture ?

| Principe | Explication |
|---|---|
| **Séparation des responsabilités** | Chaque module a UN seul rôle clair. |
| **Indépendance** | Modifier `document_loader.py` ne casse pas `nlp_pipelines.py`. |
| **Travail en parallèle** | Amina travaille dans `core_nlp/` et `generator/`, Firdawss dans `rag_engine/`. |
| **Testabilité** | On peut tester chaque module individuellement via les notebooks. |
| **Configuration centralisée** | Tous les hyperparamètres sont dans `config.py`, pas éparpillés dans le code. |

### 3.3 Flux de Données Global (Diagramme)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FLUX DE DONNÉES DU PROJET                        │
│                                                                         │
│   PHASE 1 (NLP Classique) :                                            │
│   ─────────────────────────                                             │
│   Texte saisi par l'utilisateur                                         │
│       │                                                                 │
│       ├──→ analyze_sentiment(text)     → {"label": "POSITIVE", ...}     │
│       ├──→ answer_question(q, ctx)     → {"answer": "...", ...}         │
│       ├──→ classify_text(text)         → {"label": "...", ...}          │
│       └──→ summarize_text(text)        → {"summary": "..."}            │
│                                                                         │
│   PHASE 2 (Système RAG) :                                              │
│   ───────────────────────                                               │
│   Documents PDF dans data/corpus/                                       │
│       │                                                                 │
│       ▼                                                                 │
│   document_loader.py                                                    │
│       │ (extraction texte + chunking)                                   │
│       ▼                                                                 │
│   embedding_model.py                                                    │
│       │ (texte → vecteurs 384D)                                         │
│       ▼                                                                 │
│   vector_store.py                                                       │
│       │ (stockage dans index FAISS)                                     │
│       │                                                                 │
│       │ ← Question de l'utilisateur (encodée en vecteur)                │
│       │                                                                 │
│       ▼                                                                 │
│   search_top_k(query, k=3) → Top 3 chunks pertinents                   │
│       │                                                                 │
│       ▼                                                                 │
│   llm_generator.py                                                      │
│       │ (construit le prompt augmenté + appelle le LLM)                 │
│       ▼                                                                 │
│   Réponse finale affichée à l'utilisateur                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 📦 Dépendances & Installation

### 4.1 Fichier `requirements.txt`

Le projet dépend des bibliothèques Python suivantes :

```
transformers          # Hugging Face : modèles pré-entraînés (BERT, RoBERTa, BART, etc.)
torch                 # PyTorch : calculs mathématiques sur tenseurs (GPU/CPU)
sentence-transformers # Calcul d'embeddings sémantiques pour le RAG
faiss-cpu             # Facebook AI Similarity Search : index vectoriel ultra-rapide
streamlit             # Framework Web pour créer l'interface utilisateur interactive
pypdf                 # Extraction de texte à partir de fichiers PDF
langchain             # Framework pour découper les textes en chunks (Text Splitter)
```

### 4.2 Rôle de Chaque Bibliothèque

| Bibliothèque | Utilisée dans | Rôle |
|---|---|---|
| `transformers` | `nlp_pipelines.py` | Charger les modèles NLP (sentiment, QA, classification, résumé) |
| `torch` | `nlp_pipelines.py` | Manipuler les tenseurs, faire l'inférence mathématique |
| `sentence-transformers` | `embedding_model.py`, `vector_store.py` | Encoder du texte en vecteurs numériques |
| `faiss-cpu` | `vector_store.py` | Créer et interroger l'index de recherche vectorielle |
| `streamlit` | `app.py` | Construire l'interface Web interactive |
| `pypdf` | `document_loader.py` | Lire et extraire le texte des fichiers PDF |
| `langchain` | `document_loader.py` | Découper les longs textes en morceaux (chunks) |

### 4.3 Installation Pas à Pas

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd nlp-transformers-rag

# 2. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate         # Windows

# 3. Installer toutes les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

---

## 5. ⚙️ Phase 1 — Pipelines NLP avec Transformers

### 5.1 Fichier de Configuration : `src/config.py`

Ce fichier centralise **tous les choix de modèles et hyperparamètres** du projet. Cela évite
d'avoir des noms de modèles en dur (hardcodés) un peu partout dans le code.

```python
# src/config.py
import os

# ── Phase 1 : Modèles NLP ──
NLP_CLASSIFIER_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"
NLP_SENTIMENT_MODEL  = "distilbert-base-uncased-finetuned-sst-2-english"
NLP_QA_MODEL         = "deepset/roberta-base-squad2"
NLP_SUMMARIZER_MODEL = "facebook/bart-large-cnn"

# ── Phase 2 : Modèles RAG ──
EMBEDDING_MODEL_NAME  = "all-MiniLM-L6-v2"
GENERATIVE_LLM_MODEL  = "google/gemma-2b-it"

# ── Hyperparamètres RAG ──
CHUNK_SIZE     = 500    # Taille de chaque morceau de texte (en caractères)
CHUNK_OVERLAP  = 50     # Chevauchement entre les morceaux (évite de couper une idée)
VECTOR_DB_PATH = "data/faiss_index"   # Où sauvegarder l'index FAISS
CORPUS_DIR     = "data/corpus"        # Où sont stockés les PDF de cours
```

#### Explication des Modèles Choisis :

| Variable | Modèle | Explication |
|---|---|---|
| `NLP_CLASSIFIER_MODEL` | `distilbert-base-uncased-emotion` | DistilBERT (version légère de BERT) fine-tuné pour classer les émotions (joie, tristesse, colère, peur, surprise, amour). |
| `NLP_SENTIMENT_MODEL` | `distilbert-base-uncased-finetuned-sst-2-english` | DistilBERT fine-tuné sur le dataset SST-2 pour distinguer sentiment POSITIF / NÉGATIF. |
| `NLP_QA_MODEL` | `deepset/roberta-base-squad2` | RoBERTa (version améliorée de BERT) fine-tuné sur SQuAD 2.0 pour extraire des réponses d'un contexte. |
| `NLP_SUMMARIZER_MODEL` | `facebook/bart-large-cnn` | BART (modèle encodeur-décodeur de Facebook) fine-tuné sur CNN/DailyMail pour le résumé de texte. |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Modèle léger et rapide (6 couches) qui transforme du texte en vecteurs de 384 dimensions. |
| `GENERATIVE_LLM_MODEL` | `google/gemma-2b-it` | Gemma 2B de Google — un petit modèle de langage génératif (2 milliards de paramètres) utilisable localement. |

### 5.2 Le Fichier Principal : `src/core_nlp/nlp_pipelines.py`

Ce fichier contient toutes les fonctionnalités NLP de la Phase 1. Il est découpé en plusieurs
blocs logiques :

#### 5.2.1 Les Importations

```python
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForQuestionAnswering
from src.config import *
import torch
```

- `AutoTokenizer` : Charge le bon tokenizer selon le modèle.
- `AutoModel` : Charge un modèle générique pré-entraîné.
- `pipeline` : Outil magique qui simplifie l'utilisation des modèles.
- `AutoModelForQuestionAnswering` : Charge un modèle spécialisé pour le QA (avec les couches
  `start_logits` et `end_logits`).
- `from src.config import *` : Importe toutes les constantes de configuration.
- `torch` : PyTorch pour les calculs mathématiques sur tenseurs.

#### 5.2.2 Fonction Utilitaire : `load_pretrained_model_and_tokenizer()`

```python
def load_pretrained_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model
```

**Ce que fait cette fonction :**
1. Télécharge le tokenizer du modèle demandé depuis Hugging Face Hub.
2. Télécharge le modèle pré-entraîné lui-même (poids du réseau de neurones).
3. Retourne les deux pour qu'on puisse les utiliser.

**Note :** Cette fonction charge un `AutoModel` générique. Pour des tâches spécifiques
(QA, classification), on utilise plutôt `AutoModelForQuestionAnswering` ou `pipeline()`.

#### 5.2.3 Système de Cache Global

```python
_sentiment_pipeline = None
_qa_tokenizer = None
_qa_model = None
```

**Pourquoi un cache ?**
Charger un modèle d'IA en mémoire RAM peut prendre **5 à 30 secondes** (des centaines de
millions de paramètres à charger). Si on rechargeait le modèle à chaque appel de fonction,
l'application serait inutilisable. Grâce à ces variables globales, on ne charge qu'UNE fois.

#### 5.2.4 Fonction `init_pipelines()` — Démarrage des Modèles

```python
def init_pipelines():
    global _sentiment_pipeline, _qa_tokenizer, _qa_model

    # Mode facile : pipeline() pour le sentiment
    _sentiment_pipeline = pipeline("sentiment-analysis", model=NLP_SENTIMENT_MODEL)

    # Mode manuel : Tokenizer + Model séparés pour le QA
    _qa_tokenizer = AutoTokenizer.from_pretrained(NLP_QA_MODEL)
    _qa_model = AutoModelForQuestionAnswering.from_pretrained(NLP_QA_MODEL)
```

**Deux approches différentes sont montrées ici :**

| Approche | Utilisée pour | Avantage |
|---|---|---|
| `pipeline()` (Mode facile) | Analyse de Sentiment | 1 ligne de code, tout est automatique |
| Tokenizer + Model (Mode manuel) | Question Answering | Contrôle total sur le processus, pédagogique |

Le mode manuel montre exactement ce que `pipeline()` fait sous le capot.

#### 5.2.5 Tâche 1 — Analyse de Sentiment (Amina) ✅

```python
def analyze_sentiment(text: str) -> dict:
    if _sentiment_pipeline is None:
        init_pipelines()

    result = _sentiment_pipeline(text)
    return result[0]
```

**Fonctionnement pas à pas :**
1. Vérifie si le modèle est déjà en mémoire (`if None`). Sinon, le charge.
2. Passe le texte au pipeline de sentiment.
3. Le pipeline fait tout automatiquement :
   - Tokenise le texte → `[101, 1045, 2293, 2023, 999, 102]`
   - Passe dans DistilBERT → obtient des logits
   - Applique softmax → `[0.0002, 0.9998]` (prob NÉGATIF, prob POSITIF)
   - Retourne `{'label': 'POSITIVE', 'score': 0.9998}`

**Exemple d'utilisation :**
```python
>>> analyze_sentiment("I love this course on NLP!")
{'label': 'POSITIVE', 'score': 0.9998}

>>> analyze_sentiment("This is terrible and boring.")
{'label': 'NEGATIVE', 'score': 0.9994}
```

#### 5.2.6 Tâche 2 — Question Answering (Amina) ✅

C'est la partie la plus **pédagogique** du code. Au lieu d'utiliser `pipeline("question-answering")`,
le QA est implémenté **manuellement** pour montrer les 3 étapes (pre-processing, modèle, post-processing).

```python
def answer_question(question: str, context: str) -> dict:
    if _qa_tokenizer is None or _qa_model is None:
        init_pipelines()

    # ÉTAPE 1 : Pré-processing (Tokenisation)
    inputs = _qa_tokenizer(question, context, return_tensors="pt")

    # ÉTAPE 2 : Inférence (Passage dans le réseau de neurones)
    with torch.no_grad():
        outputs = _qa_model(**inputs)

    # ÉTAPE 3 : Post-processing (Extraction de la réponse)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    answer_tokens = inputs.input_ids[0][start_idx:end_idx]
    answer = _qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)

    score = (torch.max(outputs.start_logits) + torch.max(outputs.end_logits)).item() / 2.0
    return {"answer": answer, "score": score}
```

**Explication détaillée de chaque étape :**

**Étape 1 — Tokenisation :**
```python
inputs = _qa_tokenizer(question, context, return_tensors="pt")
```
- Le tokenizer prend la question ET le contexte et les convertit en tokens.
- Le format spécial est : `[CLS] question [SEP] contexte [SEP]`
- `return_tensors="pt"` signifie que la sortie sera des tenseurs PyTorch.

**Étape 2 — Inférence :**
```python
with torch.no_grad():
    outputs = _qa_model(**inputs)
```
- `torch.no_grad()` désactive le calcul des gradients (on n'entraîne pas, on utilise juste le modèle).
  Cela économise de la mémoire et accélère le calcul.
- `_qa_model(**inputs)` passe les tokens dans RoBERTa.
- Le modèle retourne deux séries de scores :
  - `start_logits` : un score pour chaque token indiquant la probabilité qu'il soit le DÉBUT de la réponse.
  - `end_logits` : un score pour chaque token indiquant la probabilité qu'il soit la FIN de la réponse.

**Étape 3 — Post-processing :**
```python
start_idx = torch.argmax(outputs.start_logits)   # Token avec le plus haut score de début
end_idx = torch.argmax(outputs.end_logits) + 1    # Token avec le plus haut score de fin (+1 car slice exclusive)
answer_tokens = inputs.input_ids[0][start_idx:end_idx]  # Extraire les token IDs de la réponse
answer = _qa_tokenizer.decode(answer_tokens)      # Reconvertir les IDs en texte lisible
```

**Exemple d'utilisation :**
```python
>>> context = "Le NLP est une branche de l'intelligence artificielle. Il a été révolutionné par les Transformers en 2017."
>>> answer_question("Qu'est-ce que le NLP ?", context)
{'answer': 'une branche de l\'intelligence artificielle', 'score': 7.23}
```

#### 5.2.7 Tâche 3 — Classification de Texte (Firdawss) ⏳

**Non encore implémentée.** Cette fonction doit classifier un texte dans une catégorie
prédéfinie (ex : émotion, sujet). Le modèle prévu est `distilbert-base-uncased-emotion`.

**Ce qu'il faudra coder :**
```python
def classify_text(text: str) -> dict:
    # Utiliser pipeline("text-classification", model=NLP_CLASSIFIER_MODEL)
    # Retourner : {"label": "joy", "score": 0.97}
    pass
```

#### 5.2.8 Tâche 4 — Résumé Automatique (Firdawss) ⏳

**Non encore implémentée.** Cette fonction doit résumer un texte long en un texte court.
Le modèle prévu est `facebook/bart-large-cnn`.

**Ce qu'il faudra coder :**
```python
def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> dict:
    # Utiliser pipeline("summarization", model=NLP_SUMMARIZER_MODEL)
    # Retourner : {"summary_text": "Le résumé du texte..."}
    pass
```

---

## 6. 🔍 Phase 2 — Système RAG (Retrieval-Augmented Generation)

### 6.1 Module d'Ingestion : `src/rag_engine/document_loader.py`

Ce fichier est responsable de **lire les fichiers PDF** et de **découper le texte** en
morceaux (chunks) de taille uniforme.

#### Code actuel (squelette) :

```python
from typing import List, Dict
import os
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf_documents(folder_path: str) -> List[Dict[str, str]]:
    """
    Charger tous les documents PDF du dossier corpus.
    Retourne : [{"text": "...", "source": "nom_fichier.pdf"}]
    """
    pass  # À implémenter par Firdawss

def get_chunks(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Découper les documents chargés en chunks homogènes.
    """
    pass  # À implémenter par Firdawss
```

#### Ce que ces fonctions doivent faire :

**`load_pdf_documents(folder_path)` :**
1. Parcourir tous les fichiers `.pdf` dans le dossier `data/corpus/`.
2. Pour chaque PDF, utiliser `pypdf` (ou LangChain `PyPDFLoader`) pour extraire le texte.
3. Retourner une liste de dictionnaires avec le texte et le nom du fichier source.

**`get_chunks(documents)` :**
1. Prendre les documents extraits et les découper en morceaux de `CHUNK_SIZE` caractères (500).
2. Chaque morceau se chevauche avec le précédent de `CHUNK_OVERLAP` caractères (50).
3. Le chevauchement évite de couper une phrase ou une idée au milieu.

```
Document original : "AAAA AAAA BBBB BBBB CCCC CCCC DDDD DDDD"
                     |-- chunk 1 --|
                              |-- chunk 2 --|
                                       |-- chunk 3 --|
                     ← overlap →
```

### 6.2 Module d'Embeddings : `src/rag_engine/embedding_model.py`

Ce fichier fournit un wrapper autour du modèle `sentence-transformers` pour calculer des
embeddings sémantiques.

#### Code actuel (implémenté) :

```python
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
```

#### Explication :

- **`SentenceTransformer(model_name)`** : Charge le modèle `all-MiniLM-L6-v2` en mémoire.
  Ce modèle est basé sur un Transformer BERT miniaturisé (6 couches au lieu de 12) et
  a été fine-tuné pour que les phrases sémantiquement proches aient des vecteurs proches.

- **`self.model.encode(texts)`** : Prend une liste de textes et retourne une matrice NumPy
  de dimensions `(nombre_de_textes, 384)`. Chaque ligne est l'embedding d'un texte.

- **`.tolist()`** : Convertit la matrice NumPy en une liste Python native.

**Exemple :**
```python
wrapper = EmbeddingModelWrapper()
embeddings = wrapper.embed_texts(["Bonjour le monde", "Hello world"])
# embeddings[0] → [0.12, -0.34, 0.56, ...]  (384 nombres)
# embeddings[1] → [0.11, -0.32, 0.58, ...]  (similaire car sens proche !)
```

### 6.3 Module de Recherche Vectorielle : `src/rag_engine/vector_store.py`

Ce fichier gère la **base de données vectorielle FAISS** — le cœur du système RAG.

#### Code actuel (implémenté) :

```python
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
        self.documents = []  # [{id, text, source}]

    def build_and_save_index(self, chunks: List[Dict[str, str]]):
        self.documents = chunks
        texts = [c["text"] for c in chunks]

        # 1. Calculer les embeddings de tous les chunks
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        dimension = embeddings.shape[1]  # 384

        # 2. Créer l'index FAISS (recherche par distance L2)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

        print(f"✅ Index FAISS créé avec {self.index.ntotal} documents.")

    def search_top_k(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        if self.index is None:
            raise ValueError("L'index FAISS n'est pas initialisé !")

        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(
            np.array(query_vector).astype('float32'), k
        )

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results
```

#### Explication détaillée :

**`__init__()` — Initialisation :**
- Charge le modèle d'embeddings (`SentenceTransformer`).
- Prépare un espace pour l'index FAISS et la liste des documents.

**`build_and_save_index(chunks)` — Construction de l'Index :**
1. Reçoit les chunks de texte (sortie de `document_loader.py`).
2. Encode chaque chunk en un vecteur de 384 dimensions.
3. Crée un index FAISS de type `IndexFlatL2` (recherche exacte par distance L2/euclidienne).
4. Ajoute tous les vecteurs dans l'index.

**`search_top_k(query, k)` — Recherche Sémantique :**
1. Encode la question de l'utilisateur en un vecteur.
2. Cherche dans FAISS les `k` vecteurs les plus proches (distance L2 minimale).
3. FAISS retourne les indices et les distances des résultats.
4. On retrouve les textes originaux correspondants dans `self.documents`.

**Visualisation de la recherche FAISS :**
```
Index FAISS (espace 384D simplifié en 2D) :

    •chunk_15     •chunk_42
        •chunk_3
                    ★ ← Vecteur de la question
        •chunk_8
    •chunk_21         •chunk_7

Top-3 les plus proches de ★ : chunk_3, chunk_8, chunk_42
```

### 6.4 Module de Génération : `src/generator/llm_generator.py`

Ce fichier est responsable de **construire le prompt augmenté** et d'**appeler le modèle
génératif** (LLM) pour produire la réponse finale.

#### Code actuel (squelette fonctionnel) :

```python
from typing import List, Dict
from src.config import GENERATIVE_LLM_MODEL

class RAGGenerator:
    def __init__(self, model_name_or_api_key: str = GENERATIVE_LLM_MODEL):
        pass  # Amina : Initialiser le LLM ici

    def build_prompt(self, query: str, contexts: List[Dict[str, str]]) -> str:
        combined_context = "\n\n".join([
            f"[Source: {c['source']}]\n{c['text']}"
            for c in contexts
        ])

        prompt = f"""Vous êtes un assistant IA pédagogique et rigoureux.
Répondez de manière structurée et claire à la question posée
en vous basant uniquement sur le contexte ci-dessous.
Si la réponse ne figure pas dans le contexte fourni,
dites simplement : "Je ne trouve pas la réponse dans les documents fournis."

Contexte extrait du cours :
---------------------
{combined_context}
---------------------

Question de l'étudiant : {query}

Réponse claire et détaillée :"""
        return prompt

    def generate(self, query: str, contexts: List[Dict[str, str]]) -> str:
        prompt = self.build_prompt(query, contexts)
        # Amina : Appeler le LLM ici
        return "Réponse générée (à connecter au modèle LLM d'Amina)"
```

#### Explication du Prompt Engineering :

Le prompt est structuré de manière précise pour **guider** le LLM :

1. **Rôle** : On dit au LLM qu'il est un "assistant pédagogique rigoureux" pour cadrer ses réponses.
2. **Instruction** : On lui dit de ne répondre **qu'à partir du contexte fourni**.
3. **Garde-fou** : Si la réponse n'est pas dans le contexte, il doit le dire explicitement
   (au lieu d'inventer).
4. **Contexte** : Les chunks pertinents trouvés par FAISS, avec leurs sources.
5. **Question** : La question de l'utilisateur.

**Pourquoi cette structure ?**
- Sans le garde-fou, le LLM pourrait **halluciner** des réponses fausses.
- Les sources permettent à l'utilisateur de vérifier la réponse.
- Le format structuré aide le LLM à mieux comprendre ce qu'on attend de lui.

**Ce qu'il reste à faire :**
- Connecter un vrai LLM (ex : API Gemini, modèle local Gemma, ou API HuggingFace Inference).
- Remplacer le `return` statique par l'appel réel au modèle.

---

## 7. 🖥️ Interface Utilisateur Streamlit

### 7.1 Fichier `app.py` — Structure Actuelle

L'interface Streamlit est organisée en **deux onglets** :

```python
import streamlit as st

st.set_page_config(
    page_title="ENSA NLP & RAG Tool",
    page_icon="🚀",
    layout="wide"
)

st.title("📚 Mini-Projet : Transformers & Systèmes RAG")
st.write("Bienvenue dans votre plateforme d'apprentissage NLP interactive.")

tab1, tab2 = st.tabs([
    "⚙️ Boîte à outils NLP (Phase 1)",
    "💬 Chatbot RAG Intelligent (Phase 2)"
])

with tab1:
    st.header("Phase 1 : Exploration NLP")
    st.write("Testez les tâches NLP implémentées par Amina et Firdawss.")
    # TODO : Formulaires pour tester les 4 pipelines

with tab2:
    st.header("Phase 2 : Système RAG")
    st.write("Posez des questions sur vos cours stockés dans data/corpus/.")
    # TODO : Upload de PDF, chatbot, comparaison avec/sans RAG
```

### 7.2 Ce que l'Interface Finale Devra Contenir

**Onglet 1 — Boîte à outils NLP :**
- Un champ de texte pour saisir une phrase.
- 4 boutons : Sentiment, Classification, QA, Résumé.
- Affichage des résultats avec les scores et les labels.

**Onglet 2 — Chatbot RAG :**
- Un bouton pour uploader des fichiers PDF.
- Un champ de saisie pour poser une question.
- L'affichage de la réponse générée avec les sources citées.
- Une comparaison côte-à-côte : réponse **avec RAG** vs **sans RAG**.

---

## 8. 🤝 Répartition des Tâches entre Amina & Firdawss

### 8.1 Vue Synthétique

```
┌──────────────────────────────────────────────────────────────────┐
│                    RÉPARTITION DES TÂCHES                        │
│                                                                  │
│  🟦 AMINA                          🟫 FIRDAWSS                  │
│  ──────────                        ──────────                    │
│                                                                  │
│  Phase 1 :                         Phase 1 :                     │
│  ✅ Analyse de Sentiment           ⏳ Classification de Texte    │
│  ✅ Question Answering             ⏳ Résumé Automatique         │
│  ✅ Architecture & config.py       ⏳ Notebook de Validation     │
│                                                                  │
│  Phase 2 :                         Phase 2 :                     │
│  ⏳ Prompt Engineering (LLM)       ⏳ Chargement PDF (Loader)    │
│  ⏳ Connexion API/LLM              ⏳ Embeddings & FAISS         │
│  ⏳ Interface Streamlit             ⏳ search_top_k()             │
│                                                                  │
│  🟪 TRAVAIL COMMUN :                                            │
│  ⏳ Tests comparatifs (avec/sans RAG)                            │
│  ⏳ Rédaction du rapport final                                   │
│  ⏳ Démonstration                                                │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 Contrat d'Interface

Pour que les modules d'Amina et Firdawss s'assemblent correctement, les **signatures de
fonctions** suivantes ont été convenues :

| Fonction | Module | Entrée | Sortie |
|---|---|---|---|
| `analyze_sentiment(text)` | `nlp_pipelines.py` | `str` | `{"label": str, "score": float}` |
| `answer_question(q, ctx)` | `nlp_pipelines.py` | `str, str` | `{"answer": str, "score": float}` |
| `classify_text(text)` | `nlp_pipelines.py` | `str` | `{"label": str, "score": float}` |
| `summarize_text(text)` | `nlp_pipelines.py` | `str` | `{"summary_text": str}` |
| `load_pdf_documents(path)` | `document_loader.py` | `str` | `[{"text": str, "source": str}]` |
| `get_chunks(docs)` | `document_loader.py` | `List[Dict]` | `[{"text": str, "source": str}]` |
| `build_and_save_index(chunks)` | `vector_store.py` | `List[Dict]` | `None` |
| `search_top_k(query, k)` | `vector_store.py` | `str, int` | `[{"text": str, "source": str}]` |
| `build_prompt(query, ctxs)` | `llm_generator.py` | `str, List[Dict]` | `str` |
| `generate(query, ctxs)` | `llm_generator.py` | `str, List[Dict]` | `str` |

---

## 9. 📊 État d'Avancement Actuel

### 9.1 Tableau Récapitulatif

| Module | Fichier | État | Développeur |
|---|---|---|---|
| Configuration | `src/config.py` | ✅ **Terminé** | Amina |
| Analyse de Sentiment | `src/core_nlp/nlp_pipelines.py` | ✅ **Terminé** | Amina |
| Question Answering | `src/core_nlp/nlp_pipelines.py` | ✅ **Terminé** | Amina |
| Classification de Texte | `src/core_nlp/nlp_pipelines.py` | ❌ **Non commencé** | Firdawss |
| Résumé Automatique | `src/core_nlp/nlp_pipelines.py` | ❌ **Non commencé** | Firdawss |
| Chargement PDF | `src/rag_engine/document_loader.py` | 🟡 **Squelette seulement** | Firdawss |
| Embeddings | `src/rag_engine/embedding_model.py` | ✅ **Terminé** | Firdawss |
| Index FAISS | `src/rag_engine/vector_store.py` | ✅ **Terminé** | Firdawss |
| Prompt Engineering | `src/generator/llm_generator.py` | 🟡 **Prompt prêt, LLM non connecté** | Amina |
| Interface Streamlit | `app.py` | 🟡 **Structure de base seulement** | Amina |
| Notebook Phase 1 | `notebooks/01_exploration_transformers.ipynb` | 🟡 **Créé** | Firdawss |
| Notebook Phase 2 | `notebooks/02_experimentation_rag.ipynb` | ❌ **Vide** | — |

### 9.2 Pourcentage de Complétion Estimé

```
Phase 1 (Pipelines NLP) :    ████████░░░░░░░░  ~50%  (2/4 tâches terminées)
Phase 2 (Système RAG) :      ██████░░░░░░░░░░  ~40%  (Embeddings + FAISS ok, Loader + LLM en attente)
Interface Streamlit :         ██░░░░░░░░░░░░░░  ~15%  (Structure seulement)
────────────────────────────────────────────────────
TOTAL PROJET :                █████░░░░░░░░░░░  ~35%
```

---

## 10. 🚀 Prochaines Étapes & Travail Restant

### 10.1 Priorité Haute (Phase 1 — à terminer en premier)

| # | Tâche | Responsable | Détail |
|---|---|---|---|
| 1 | Implémenter `classify_text()` | Firdawss | Utiliser `pipeline("text-classification", model=NLP_CLASSIFIER_MODEL)` |
| 2 | Implémenter `summarize_text()` | Firdawss | Utiliser `pipeline("summarization", model=NLP_SUMMARIZER_MODEL)` |
| 3 | Valider dans le Notebook Phase 1 | Firdawss | Tester les 4 fonctions avec des exemples variés |

### 10.2 Priorité Moyenne (Phase 2 — RAG)

| # | Tâche | Responsable | Détail |
|---|---|---|---|
| 4 | Implémenter `load_pdf_documents()` | Firdawss | Utiliser `pypdf.PdfReader` pour extraire le texte page par page |
| 5 | Implémenter `get_chunks()` | Firdawss | Utiliser `langchain.text_splitter.RecursiveCharacterTextSplitter` |
| 6 | Connecter le LLM dans `generate()` | Amina | Option A : API Gemini / Option B : Hugging Face Pipeline local |
| 7 | Sauvegarder/charger l'index FAISS | Firdawss | Utiliser `faiss.write_index()` et `faiss.read_index()` |

### 10.3 Priorité Basse (Intégration & Livraison)

| # | Tâche | Responsable | Détail |
|---|---|---|---|
| 8 | Développer l'interface Streamlit complète | Amina | Formulaires, boutons, affichage des résultats |
| 9 | Comparaison avec vs sans RAG | Amina & Firdawss | Envoyer la même question avec et sans contexte |
| 10 | Rédaction du rapport PDF final | Amina & Firdawss | Inclure l'architecture, le diagramme, les résultats, les captures d'écran |

---

## 📎 Annexe : Fichiers du Projet avec leurs Chemins

| Fichier | Chemin Complet | Taille |
|---|---|---|
| `config.py` | `src/config.py` | 761 octets |
| `nlp_pipelines.py` | `src/core_nlp/nlp_pipelines.py` | 3 623 octets |
| `document_loader.py` | `src/rag_engine/document_loader.py` | 751 octets |
| `embedding_model.py` | `src/rag_engine/embedding_model.py` | 759 octets |
| `vector_store.py` | `src/rag_engine/vector_store.py` | 1 986 octets |
| `llm_generator.py` | `src/generator/llm_generator.py` | 1 769 octets |
| `app.py` | `app.py` | 1 275 octets |
| `requirements.txt` | `requirements.txt` | 77 octets |

---

> **Dernière mise à jour** : 1er Juin 2026
