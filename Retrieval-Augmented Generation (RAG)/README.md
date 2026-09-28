# Retrieval-Augmented Generation (RAG)

This folder contains a few practical implementations of **Retrieval-Augmented Generation (RAG)** using different retrieval backends and embedding models.

The goal is mainly to experiment with the RAG pipeline:

**data → embeddings → retrieval → LLM response**

Most examples use structured data such as Excel files, with an additional example using MySQL.

## Implementations

| File | Retrieval / Embedding | Data source |
|---|---|---|
| [`Fichier-excel-Chroma.py`](./Fichier-excel-Chroma.py) | ChromaDB | Excel |
| [`Fichier-excel-FAISS.py`](./Fichier-excel-FAISS.py) | FAISS | Excel |
| [`Fichier-excel-Qdrant.py`](./Fichier-excel-Qdrant.py) | Qdrant | Excel |
| [`Qdrant-GPT-Excel.py`](./Qdrant-GPT-Excel.py) | Qdrant + GPT | Excel |
| [`Sentence-Bert-Embedding-RAG.py`](./Sentence-Bert-Embedding-RAG.py) | Qdrant + Sentence-BERT | Text / structured data |
| [`Mysql-RAG-GPT.py`](./Mysql-RAG-GPT.py) | GPT-based retrieval pipeline | MySQL |

## What is being compared?

The examples mainly explore three choices in a RAG system:

- **Vector storage / retrieval:** ChromaDB, FAISS and Qdrant
- **Embeddings:** OpenAI embeddings and Sentence-BERT
- **Generation:** GPT-based language models

The workflow is similar across the scripts: load the data, create embeddings, index them, retrieve the most relevant information for a query, and pass the retrieved context to the language model.

## Notes

These scripts are small experiments rather than a production RAG framework. They are useful for comparing different retrieval approaches and understanding how the individual components of a RAG pipeline fit together.
