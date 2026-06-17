import os

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

import glob

output_dir = r"c:\Users\amina\OneDrive\Bureau\ID\ID2\S2\NLP\projet\nlp-transformers-rag\data\corpus"

# Supprimer les anciens PDF
for old_pdf in glob.glob(os.path.join(output_dir, "*.pdf")):
    os.remove(old_pdf)
    print(f"Supprimé : {os.path.basename(old_pdf)}")

docs = [
    {
        "filename": "01_Introduction_NLP.pdf",
        "title": "Chapitre 1 : Introduction au Traitement du Langage Naturel (NLP)",
        "content": [
            "Le Traitement du Langage Naturel (NLP), ou Natural Language Processing en anglais, est un domaine pluridisciplinaire impliquant la linguistique, l'informatique et l'intelligence artificielle.",
            "L'objectif principal du NLP est de fournir aux machines la capacité de comprendre, d'interpréter, de manipuler et de générer du langage humain de manière utile et pertinente.",
            "Historiquement, les premières approches du NLP reposaient sur des systèmes symboliques et des règles grammaticales écrites à la main par des linguistes. Bien que ces systèmes aient eu du succès dans des contextes très limités, ils étaient incapables de gérer la complexité, l'ambiguïté et les nuances infinies du langage naturel.",
            "Dans les années 1990, l'introduction des méthodes statistiques a révolutionné le domaine. Les modèles de Machine Learning, comme les machines à vecteurs de support (SVM) ou les modèles de Markov cachés (HMM), ont permis aux systèmes d'apprendre des motifs à partir de grands corpus de textes, sans nécessiter de règles explicites.",
            "Aujourd'hui, l'approche dominante utilise le Deep Learning, en particulier les réseaux de neurones récurrents (RNN) et, plus récemment, les Transformers. Ces technologies ont permis des percées spectaculaires dans de nombreuses applications.",
            "Les applications du NLP sont omniprésentes dans notre vie quotidienne. Elles incluent la traduction automatique (comme Google Translate), l'analyse de sentiment pour évaluer les avis clients sur internet, la reconnaissance vocale (comme Siri ou Alexa), et bien sûr les agents conversationnels ou chatbots (comme ChatGPT).",
            "Cependant, le NLP doit encore faire face à de nombreux défis. La compréhension du sarcasme, de l'ironie, des expressions idiomatiques, ou encore des références culturelles reste très difficile pour les algorithmes actuels. De plus, les modèles requièrent des quantités massives de données et de puissance de calcul pour fonctionner correctement."
        ]
    },
    {
        "filename": "02_Les_Transformers.pdf",
        "title": "Chapitre 2 : L'Architecture Transformer et la Révolution de l'IA",
        "content": [
            "L'architecture Transformer a été dévoilée au monde par une équipe de chercheurs de Google en 2017 dans l'article historique intitulé 'Attention is All You Need'. Cette découverte a provoqué un véritable séisme dans le domaine du Deep Learning et du NLP.",
            "Avant les Transformers, les architectures dominantes pour traiter les séquences de texte étaient les Réseaux de Neurones Récurrents (RNN) et les réseaux LSTM. Le problème majeur de ces anciennes architectures était leur nature séquentielle : pour comprendre le 10ème mot d'une phrase, le réseau devait d'abord calculer les 9 mots précédents un par un.",
            "L'innovation fondamentale du Transformer est le mécanisme de 'Self-Attention' (Auto-Attention). Ce mécanisme permet au modèle de regarder l'intégralité de la phrase d'un seul coup et d'évaluer mathématiquement quelles parties de la phrase sont connectées entre elles, peu importe leur distance dans le texte.",
            "Le fait que le Transformer n'ait pas besoin de lire les mots dans l'ordre de manière séquentielle lui confère un avantage énorme : il est massivement parallélisable. Cela signifie qu'on peut l'entraîner sur des milliers de cartes graphiques (GPU) simultanément. C'est ce qui a permis de créer des modèles gigantesques entraînés sur des portions massives de l'internet.",
            "Deux familles principales ont émergé de cette architecture. D'une part, les modèles de type Encodeur comme BERT (Bidirectional Encoder Representations from Transformers), qui sont excellents pour comprendre le contexte, classifier du texte, ou extraire des réponses. D'autre part, les modèles de type Décodeur comme GPT (Generative Pre-trained Transformer), qui sont spécialisés dans la génération de texte de manière auto-régressive."
        ]
    },
    {
        "filename": "03_Systemes_RAG.pdf",
        "title": "Chapitre 3 : RAG (Retrieval-Augmented Generation)",
        "content": [
            "Le RAG, ou Retrieval-Augmented Generation (Génération Augmentée par la Recherche), est un paradigme novateur conçu pour pallier les défauts inhérents des grands modèles de langage (LLM).",
            "Malgré leur impressionnante capacité à générer du texte fluide, les LLM classiques souffrent de plusieurs problèmes critiques. Ils ont une date de coupure de leurs connaissances (ils ignorent ce qui s'est passé après leur entraînement), ils ne peuvent pas accéder à des données privées ou confidentielles d'une entreprise, et surtout, ils sont sujets aux 'hallucinations' : ils peuvent affirmer avec aplomb des faits totalement faux.",
            "L'architecture RAG résout ces problèmes en introduisant une étape de recherche (Retrieval) avant la génération de texte. Le processus se divise en plusieurs phases clés :",
            "1. L'indexation : Les documents de l'entreprise (PDF, pages web, bases de données) sont découpés en petits morceaux (chunks). Chaque morceau est transformé en un vecteur mathématique (embedding) grâce à un modèle NLP d'encodage, puis stocké dans une base de données vectorielle spécialisée comme FAISS, ChromaDB ou Pinecone.",
            "2. La recherche : Lorsqu'un utilisateur pose une question, cette question est également transformée en vecteur. La base de données recherche alors les morceaux de texte dont les vecteurs sont mathématiquement les plus proches (similarité cosinus) de la question.",
            "3. La génération : Les morceaux de texte retrouvés (le contexte) sont fusionnés avec la question originale de l'utilisateur pour former un super-prompt. Ce prompt est envoyé au LLM avec une consigne stricte du type : 'Réponds à la question en utilisant EXCLUSIVEMENT le contexte fourni ci-dessous'.",
            "Cette méthode combine le meilleur des deux mondes : la précision absolue d'un moteur de recherche documentaire et l'intelligence conversationnelle d'un LLM. Le RAG est aujourd'hui devenu le standard de l'industrie pour déployer des applications d'IA générative en entreprise."
        ]
    }
]

styles = getSampleStyleSheet()

for doc_data in docs:
    filepath = os.path.join(output_dir, doc_data["filename"])
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    story = []
    
    story.append(Paragraph(doc_data["title"], styles['Title']))
    story.append(Spacer(1, 12))
    
    for paragraph_text in doc_data["content"]:
        story.append(Paragraph(paragraph_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
    doc.build(story)
    print(f"Généré : {doc_data['filename']}")
