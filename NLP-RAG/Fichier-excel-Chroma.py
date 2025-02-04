from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os
from langchain.schema import Document
from langchain_chroma import Chroma
import pandas as pd

# Chargement des variables d'environnement
load_dotenv()

def init_chat():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", temperature=0.5)
 # Charger le fichier Excel
    content = pd.read_excel("Votre_fichier.xlsx")
    documents = []
    for index, row in content.iterrows():
        question = str(row['question'])  # Assurez-vous que les colonnes s'appellent 'question' et 'réponse'
        answer = str(row['réponse'])
        text = f"Question: {question}  {question}  {question}  {question}  Réponse: {answer}"
        documents.append(Document(page_content=text))
    
    # Répertoire pour le stockage des documents
    faq_path = "faq2"
    
    # Création d'un magasin de données avec Chroma
    data_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=faq_path)
    data_store2 = Chroma(persist_directory=faq_path, embedding_function=embeddings)
    data_store.get()
    
    # Créer un retriever pour la récupération des documents
    retriever = data_store2.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7})
    #retriever = data_store2.as_retriever(search_type="similarity", search_kwargs={"k": 70})

    
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

    prompt = PromptTemplate(
        template=template
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain



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



