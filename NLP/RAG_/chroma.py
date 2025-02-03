# Import Dependencies
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import pandas as pd
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Charger les variables d’environnement
load_dotenv()

# Initialiser les embeddings et le modèle LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", temperature=0.5, timeout=120)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Charger le fichier Excel avec pandas
content = pd.read_excel("padme.xlsx")
documents = [
    Document(page_content=f"Question: {str(row['question']) * 4} Réponse: {str(row['réponse'])}")
    for _, row in content.iterrows()
]

# Créer et sauvegarder la base de connaissances avec FAISS
DB_FAISS_PATH = 'vectorstore/db_faiss_s'
vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
vectorstore.save_local(DB_FAISS_PATH)

# Définition du template
template = """
    En tant qu’assistant virtuel BFT, tu réponds aux questions des agents de terrain, caissiers et
    gestionnaires d’agences sur les procédures, problèmes techniques ou services de BFT.
    Important : Les réponses doivent être précises et vérifiées pour éviter toute erreur et ne réponds
    que lorsque tu es sûr de la similitude avec une question précédente.
    
    Procédure :
    1- Pour les salutations ou remerciements, réponds respectueusement.
    2- Si le contexte est fourni ("{context}" n'est pas vide), utilise-le pour formuler une réponse personnalisée à "{input}".
    3- Si le contexte est vide, réponds : "Veuillez clarifier votre question."
    4- Historique et personnalisation : Adapte ta réponse selon l'historique et le ton de l'utilisateur.
    
    Historique: {history}
    Question : {input}
    Contexte : {context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=['input', 'history', 'context']  # Correction : ajout de 'context'
)

# Fonction pour charger la base de connaissances
def load_knowledge_base():
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Fonction pour formater les documents en texte
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# Fonction principale d’interaction avec l’IA
def Talker(query: str, history):
    retriever = load_knowledge_base().as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
    )

    # Récupérer le contexte pertinent
    retrieved_docs = retriever.get_relevant_documents(query)
    context = format_docs(retrieved_docs) if retrieved_docs else ""

    # Construire la chaîne de traitement
    rag_chain = (
        {"context": RunnablePassthrough(), "input": RunnablePassthrough(), "history": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Générer la réponse
    output = rag_chain.invoke({"context": context, "input": query, "history": history})

    return output
