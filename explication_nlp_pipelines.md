# 🧠 Comprendre le code : `nlp_pipelines.py`

Ce document explique de manière simple et détaillée comment fonctionne l'outil `pipeline` de Hugging Face et analyse ligne par ligne les fonctionnalités de votre fichier `nlp_pipelines.py`.

---

## 1. 🪄 Comment fonctionne la fonction `pipeline` (De manière simple)

Imaginez que vous voulez traduire un texte. Normalement, vous devez :
1. Chercher un dictionnaire.
2. Découper votre phrase mot par mot.
3. Chercher la traduction de chaque mot.
4. Reconstruire la phrase avec la bonne grammaire.

C'est fastidieux ! Et bien, dans l'Intelligence Artificielle, c'est pareil. Pour qu'un modèle comprenne un texte, il faut 3 étapes complexes :
1. **Le Pre-processing (Tokenization) :** Découper le texte humain en nombres mathématiques.
2. **Le Model (Inférence) :** Faire passer ces nombres dans le réseau de neurones complexe.
3. **Le Post-processing :** Reconvertir les probabilités mathématiques du réseau en une réponse humaine compréhensible (comme "POSITIF" ou "NÉGATIF").

👉 **Le rôle de la fonction `pipeline()` :**
La fonction `pipeline` de Hugging Face est une fonction "magique" qui **cache ces 3 étapes complexes**. Vous lui donnez juste la tâche (ex: `"sentiment-analysis"`) et le texte, et elle fait tout le travail mathématique en coulisse pour vous rendre directement le résultat final. C'est le moyen le plus simple d'utiliser l'IA !

---

## 2. 🔍 Explication détaillée du fichier `nlp_pipelines.py`

Voici l'explication de chaque bloc de code de votre fichier.

### A. Les Importations (Lignes 14-16)
```python
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForQuestionAnswering
from src.config import *
import torch
```
*   **`AutoTokenizer`, `AutoModel`...** : Ce sont les outils téléchargés depuis Hugging Face.
*   **`pipeline`** : L'outil "magique" expliqué ci-dessus.
*   **`import torch`** : La bibliothèque mathématique de PyTorch utilisée pour manipuler les nombres et les tenseurs en coulisse.

### B. Le Système de Cache (Lignes 27-30)
```python
_sentiment_pipeline = None
_qa_tokenizer = None
_qa_model = None
```
*   **Pourquoi ?** Charger un modèle d'Intelligence Artificielle en mémoire prend du temps (plusieurs secondes). On déclare ces variables au début (avec `None`) pour pouvoir stocker les modèles en mémoire (cache). Ainsi, si on analyse 100 phrases d'affilée, on ne télécharge le modèle qu'une seule fois !

### C. La fonction `init_pipelines()` (Ligne 32)
```python
def init_pipelines():
    global _sentiment_pipeline, _qa_tokenizer, _qa_model
    # 1. On charge l'Analyse de Sentiment avec l'outil facile "pipeline"
    _sentiment_pipeline = pipeline("sentiment-analysis", model=NLP_SENTIMENT_MODEL)
    
    # 2. On charge le Question Answering manuellement
    _qa_tokenizer = AutoTokenizer.from_pretrained(NLP_QA_MODEL)
    _qa_model = AutoModelForQuestionAnswering.from_pretrained(NLP_QA_MODEL)
```
*   Cette fonction "démarre" le moteur.
*   Elle utilise le mode facile (`pipeline`) pour l'Analyse de Sentiment.
*   Elle utilise le mode manuel (Tokenizer + Model séparés) pour le Question Answering afin d'éviter les bugs liés aux versions, et pour vous apprendre comment ça marche "sous le capot".

### D. La fonction `analyze_sentiment(text)` (Ligne 49)
```python
def analyze_sentiment(text: str) -> dict:
    if _sentiment_pipeline is None:
        init_pipelines()
    
    result = _sentiment_pipeline(text)
    return result[0]
```
*   **Comment ça marche ?** Si le pipeline n'a pas encore été chargé, on appelle `init_pipelines()`. Ensuite, on donne tout simplement le `text` à notre `_sentiment_pipeline`. Il s'occupe de tout et nous renvoie le résultat (ex: `{'label': 'POSITIVE', 'score': 0.99}`).

### E. La fonction `answer_question(question, context)` (Ligne 60)
*C'est ici que l'on voit ce que cache vraiment la fonction `pipeline` !*

```python
    # 1. Tokenisation (Pre-processing)
    inputs = _qa_tokenizer(question, context, return_tensors="pt")
```
*   Le modèle mathématique ne comprend pas les mots. On utilise le Tokenizer pour convertir votre question et le contexte en une matrice de nombres (les `tensors` de PyTorch, "pt").

```python
    # 2. Le Modèle (Inférence mathématique)
    with torch.no_grad():
        outputs = _qa_model(**inputs)
```
*   On fait passer nos nombres dans le cerveau mathématique de l'IA (le modèle RoBERTa). Il calcule les probabilités pour trouver à quel mot commence la réponse (`start_logits`) et à quel mot elle se termine (`end_logits`).

```python
    # 3. Post-processing (Trouver la réponse)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
```
*   `torch.argmax` sert à trouver le mot qui a le plus haut score de probabilité d'être le début, et pareil pour la fin.

```python
    # 4. Traduire les nombres en texte humain
    answer_tokens = inputs.input_ids[0][start_idx:end_idx]
    answer = _qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
```
*   On récupère les nombres qui correspondent à la réponse, et on demande au Tokenizer de les "décoder" (les retraduire en texte humain).

---
**En résumé :** L'Analyse de Sentiment dans votre code est la version "touriste" (facile grâce au `pipeline`), tandis que le Question Answering est la version "ingénieur" (vous gérez les tenseurs et les probabilités mathématiques vous-même !).
