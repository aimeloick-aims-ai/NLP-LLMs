from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
#from langchain.vectorstores import Qdrant
from langchain_qdrant import Qdrant
from langchain.schema import Document
import pandas as pd
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
# Chargement des variables d'environnement
load_dotenv()


def init_chat():
    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", temperature=0.5)
    #embeddings =OllamaEmbeddings()
    #######################"
    # Connexion à Qdrant
    client = QdrantClient(url="http://localhost:6333")  # Utiliser le bon endpoint
    collection_name = "BaseQdrant"

    # Vérifier si la collection existe déjà
    if not client.get_collections().collections or collection_name not in [c.name for c in client.get_collections().collections]:
        # Créer la collection si elle n'existe pas
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        # Charger et ajouter des documents initiaux si la collection n'existe pas encore
        content = pd.read_excel("votre_fichier.xlsx")
        memory = ConversationBufferMemory(memory_key="history", input_key="question")
        documents = []
        for index, row in content.iterrows():
            question = str(row['question'])
            answer = str(row['réponse'])
            # J'ai voulu plus pondérer les questions que les réponses
            text = f"Question: {question}  {question}  {question} {question} Réponse: {answer}"
            documents.append(Document(page_content=text))
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        #docs = text_splitter.split_documents(documents=documents)
        docs=documents
        vectorstore = Qdrant(
            client=client, 
            collection_name=collection_name, 
            embeddings=embeddings
        )
        vectorstore.add_documents(docs)
    else:
        # Charger l'existant si la collection est déjà là
        vectorstore = Qdrant(
            client=client, 
            collection_name=collection_name, 
            embeddings=embeddings
        )
    
    # Créer un retriever pour la récupération des documents
    #retriever = data_store2.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})
    #retriever = data_store2.as_retriever(search_type="similarity", search_kwargs={"k": 70})
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})
        # Modèle de prompt pour l'assistant
    template = """
    En tant qu’assistant virtuel, tu réponds aux questions des agents de terrain, caissiers et gestionnaires d’agences sur les procédures, problèmes techniques ou services liés à leur travail.
    Important : Les réponses doivent être précises et vérifiées pour éviter toute erreur. Ne réponds que lorsque tu es sûr de la similitude avec une question précédente.
    
    Procédure :
    Pour les salutations ou remerciements, réponds respectueusement.
    Si le contexte est fourni ("{context}" n'est pas vide), utilise-le pour formuler une réponse personnalisée à "{input}".
    Si le contexte est vide, réponds : "Veuillez clarifier votre question."
    Historique et personnalisation : Adapte ta réponse selon l'historique et le ton de l'utilisateur.
    Historique : {history}
    Question : {input}
    Contexte : {context}
    """
    # Modèle de prompt pour l'assistant
    
    prompt = PromptTemplate(template=template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain



#Historique est Optionnelle
def Talker(query:str, history):
    conversations= init_chat()
    # Préparer l'entrée sous forme de dictionnaire
    input_data = {
        "input": query,
        "history": history,
    }
    # Appel de la méthode invoke avec un dictionnaire
    output = conversations.invoke(input=input_data)
    response = output.get('answer')
    # Mettre à jour l'historique des messages avec la nouvelle question et réponse 
    return response


