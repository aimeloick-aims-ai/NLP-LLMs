from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from qdrant_client import QdrantClient
import mysql.connector

# Étape 1 : Configurations de base
OPENAI_API_KEY = "votre_clé_openai"
QDRANT_URL = "http://localhost:6333"  # URL de votre instance Qdrant
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'password',
    'database': 'rag_metadata'
}

# Étape 2 : Initialiser l'outil d'embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY
)

# Étape 3 : Initialiser Qdrant comme vector store
qdrant_client = QdrantClient(url=QDRANT_URL)
vector_store = Qdrant(client=qdrant_client, collection_name="qa_collection", embeddings=embeddings)

# Étape 4 : Charger ou ajouter des documents
documents = [
    {"text": "LangChain est un framework de RAG.", "metadata": {"source": "doc1"}},
    {"text": "GPT-4 est un puissant LLM développé par OpenAI.", "metadata": {"source": "doc2"}},
]
vector_store.add_texts([doc["text"] for doc in documents], [doc["metadata"] for doc in documents])

# Étape 5 : Initialiser GPT-4 via LangChain
llm = OpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# Étape 6 : Créer la chaîne RAG (question-réponse)
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Étape 7 : Exécuter des requêtes de question-réponse
query = "Quels outils sont utilisés dans un système RAG ?"
response = qa_chain.run(query)
print("Réponse :", response)

# (Facultatif) Étape 8 : Stocker ou récupérer les métadonnées dans MySQL
def store_metadata_in_mysql(metadata):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INT AUTO_INCREMENT PRIMARY KEY,
            source VARCHAR(255),
            data TEXT
        )
    """)
    for meta in metadata:
        cursor.execute("INSERT INTO metadata (source, data) VALUES (%s, %s)", (meta["source"], str(meta)))
    conn.commit()
    conn.close()

# Enregistrer les métadonnées
store_metadata_in_mysql([doc["metadata"] for doc in documents])
