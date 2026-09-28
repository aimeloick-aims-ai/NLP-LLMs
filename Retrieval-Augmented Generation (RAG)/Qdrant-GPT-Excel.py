from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Qdrant
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.schema import Document
from qdrant_client import QdrantClient
import pandas as pd
import os

# Chargement des variables d'environnement
load_dotenv()

# Initialisation de l'environnement
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "faq_documents"

# Initialisation de l'instance Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL)

# Fonction pour initialiser le système RAG
def init_chat():
    # Étape 1 : Initialiser l'outil d'embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

    # Étape 2 : Charger le document "intervention"
    content = pd.read_excel("padme.xlsx")
    documents = []
    for index, row in content.iterrows():
        question = str(row['question'])
        answer = str(row['réponse'])
        text = f"Question: {question}\nRéponse: {answer}"
        documents.append(Document(page_content=text))

    # Étape 3 : Stocker les embeddings dans Qdrant
    vector_store = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )
    vector_store.add_texts(
        texts=[doc.page_content for doc in documents],
        metadatas=[{"source": f"doc_{i}"} for i in range(len(documents))]
    )

    # Étape 4 : Configurer le modèle GPT-4
    llm = OpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

    # Étape 5 : Configurer le prompt et les chaînes
    template = """
    Tu fais partie d'un système RAG en tant qu'assistant virtuel Ben de BFT. Tu dois répondre aux questions des agents de terrain,
    caissiers et gestionnaires d'agences sur les procédures, problèmes techniques ou services de BFT.
    Important : Les réponses doivent être précises et vérifiées pour éviter toute erreur et ne réponds
    que lorsque tu es sûr de la concordance entre la question et ta réponse.

    question posé : {input}
    question-réponse trouvée : {context}
    historique : {history}
    """
    prompt = PromptTemplate(template=template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain

# Fonction pour gérer les conversations
def Talker(query: str, history):
    conversations = init_chat()
    # Préparer l'entrée sous forme de dictionnaire
    input_data = {
        "input": query,
        "history": history,
    }
    # Appel de la chaîne avec les données d'entrée
    output = conversations.invoke(input=input_data)
    response = output.get('answer')
    return response
