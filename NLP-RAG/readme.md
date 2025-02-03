# 🚀 Retrieval-Augmented Generation (RAG) 📚

Le Retrieval-Augmented Generation (RAG) est une approche avancée de l'intelligence artificielle qui combine la **génération de texte**  avec la **récupération d'informations pertinentes** 🔍 à partir d'une base de connaissances. Cela permet d'améliorer la **précision**  et la **fiabilité**  des réponses générées par les modèles de langage (**LLM**).

## ⚙️ Fonctionnement de RAG
1. **Requête utilisateur** 🗣️ : Une question ou un texte est soumis au système.
2. **Récupération d'informations** 📂 : Le système recherche les documents les plus pertinents dans une base vectorielle.
3. **Génération de réponse** 🤖 : Un modèle de langage (**LLM**) utilise ces documents comme contexte pour générer une réponse plus pertinente et précise.

## Types de fichiers pris en charge
Un système RAG peut traiter plusieurs types de fichiers, notamment :
- **Texte brut** (`.txt`)
- **Documents Word** (`.docx`)
- **PDF** (`.pdf`)
- **Pages HTML** (`.html`)
- **Formats JSON et CSV** pour des bases de données structurées.

## Segmentation du texte (Text Splitter)
Un **text splitter** permet de diviser un document en sections exploitables par le modèle. Différentes stratégies existent :
- **Basé sur le nombre de tokens** : Découpe le texte en fonction d’un nombre fixe de tokens.
- **Basé sur les sauts de ligne** : Segmente un document selon ses paragraphes.
- **Basé sur la structure** : Divise un document selon des titres et sous-titres.
- 
## 🏛️ Choix des Bases Vectorielles
Les bases vectorielles sont utilisées pour stocker et rechercher des représentations numériques des documents.

### 1️⃣ **ChromaDB** 🏗️
🔹 **Description** : Chroma est une base de données vectorielle moderne, optimisée pour l'IA et l'indexation rapide des embeddings.

✅ **Avantages** :
   -  Facilité d'utilisation avec une API simple.
   -  Optimisé pour le stockage et la recherche rapide.
   -  Intégration native avec LangChain et d'autres frameworks IA.
     
🔗 **Cas d'utilisation** : Projets nécessitant une solution légère et rapide pour la récupération d'informations.

📌 [📁 Accéder a RAG avec ChromaDB fichier excel](https://github.com/GOHOUEDE/IA/blob/main/NLP-RAG/Fichier-excel-Chroma.py)

### 2️⃣ **FAISS (Facebook AI Similarity Search)** 🚀
🔹 **Description** : Développé par Facebook AI, FAISS est une bibliothèque open-source dédiée à la recherche efficace sur de grands ensembles de données.

✅ **Avantages** :
   -  Très rapide pour la recherche approximative.
   -  Optimisé pour le traitement de grands volumes de données.
   -  Supporte des algorithmes avancés de quantification.
   - 
🔗 **Cas d'utilisation** : Systèmes nécessitant une indexation ultra-rapide pour des bases volumineuses.

📌 [📁 Accéder a RAG avec  FAISS fichier excel](https://github.com/GOHOUEDE/IA/blob/main/NLP-RAG/Fichier-excel-FAISS.py)

### 3️⃣ **Qdrant** 🏆
🔹 **Description** : Qdrant est un moteur de recherche vectorielle optimisé pour la scalabilité et la recherche en temps réel.
✅ **Avantages** :
   -  Performances élevées sur les requêtes en temps réel.
   -  Support de la recherche hybride (mots-clés + vecteurs).
   -  Facile à déployer avec Docker et Kubernetes.
🔗 **Cas d'utilisation** : Applications nécessitant une combinaison de recherche vectorielle et sémantique.

📌 [📁 Accéder au fichier Qdrant]((https://github.com/GOHOUEDE/IA/blob/main/NLP-RAG/Fichier-excel-Qdrant.py)

## 🧠 Choix des Embeddings
Les embeddings transforment les textes en vecteurs numériques pour faciliter la recherche sémantique.

### 🔹 **OpenAI Embeddings** ⚡
-  Fournit des représentations vectorielles de haute qualité.
-  Compatible avec divers modèles d'OpenAI (GPT, CLIP, etc.).

### 🔹 **Sentence Transformers (SBERT)** 🧩
-  Optimisé pour des tâches de similarité sémantique.
-  Pré-entraîné sur divers corpus pour une meilleure généralisation.

### 🔹 **Hugging Face Embeddings** 🤗
-  Large choix de modèles disponibles via la bibliothèque `transformers`.
-  Facilité d'intégration et d'adaptation selon les besoins spécifiques.


## 🤖 Choix du Modèle LLM
Le modèle de langage joue un rôle clé dans l'interprétation et la génération des réponses.

### 🔹 **GPT-4 (OpenAI)** 🚀
-  Modèle puissant avec des réponses détaillées et précises.
-  Intégration facile avec les API OpenAI.

### 🔹 **LLaMA (Meta AI)** 🦙
-  Modèle open-source performant.
-  Nécessite une infrastructure adaptée pour l'hébergement.

### 🔹 **Mistral 7B** 🌪️
-  Un modèle léger et efficace pour des tâches spécifiques.
-  Bon équilibre entre performance et coût d'inférence.

### 🔹 **Claude (Anthropic)** 🤝
-  Conçu pour des réponses alignées et sûres.
-  Spécialisé dans les interactions naturelles et conversationnelles.

📌 [📁 Accéder au fichier LLM](#)

## 🎯 Conclusion
Le choix des composants dans un système **RAG** dépend du cas d'utilisation et des exigences en termes de **rapidité**, **précision** et **scalabilité**. En combinant la bonne **base vectorielle**, les bons **embeddings** et un **LLM** adapté, il est possible de créer un système robuste et efficace pour la **récupération et la génération d’informations intelligentes**. 🚀

📌 [📁 Voir tous les fichiers du projet](#)

