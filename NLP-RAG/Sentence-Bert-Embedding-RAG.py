#Import Dependencies

#from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
# to create a new file named vectorstore in your current directory.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", temperature=0.5, timeout=120)
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
# Charger le fichier Excel avec pandas
content = pd.read_excel("padme.xlsx")
documents = []
for index, row in content.iterrows():
        question = str(row['question'])  # Assurez-vous que les colonnes s'appellent 'question' et 'réponse'
        answer = str(row['réponse'])
        text = f"Question: {question}  {question}  {question}  {question}  Réponse: {answer}"
        documents.append(Document(page_content=text))


#create a new file named vectorstore in your current directory.
if __name__=="__main__":
        DB_FAISS_PATH = 'vectorstore/db_faiss_hopess'
        vectorstore = FAISS.from_documents(documents=documents,embedding=embeddings)
        vectorstore.save_local(DB_FAISS_PATH)


#Je crée ici un template à utiliser pour questionner le modèle


template = """
    En tant qu’assistant virtuel BFT, tu réponds aux questions des agents de terrain, caissiers et
    gestionnaires d’agences sur les procédures, problèmes techniques ou services de BFT.
     Important : Les réponses doivent être précises et vérifiées pour éviter toute erreur et ne réponds
      que lorsque tu es sûr de la similitude avec une question précédente.
    Procédure :
    1- Pour les salutations,un remerciement réponds respectueusement.
    2- Questions similaires : Si context="{context}" n'est pas vide, repond à la quetion en question="{input}" avec le context ,
    reformule et donne la réponse de context de manière personnalisée.
    NB:  Vérifie la similitude avant de répondre et si input="{input}" n'est pas une salutation evite les salutation au début de ton message.
    3- Si context="{context}" est vide, repond "veuillez clarrifier votre question".
    3- Historique et Personnalisation : Adapte ta réponse en fonction des détails fournis par l'utilisateur, y compris son ton émotions.
     Rappel : Chaque réponse doit être soigneusement analysée avant d'être donnée pour éviter toute erreur ou diffusion d'informations.Tu dois suivre en ordre ces étapes 1 à 4-2.Merci

    Question : {input}
    Contexte : {context}
"""



prompt = PromptTemplate(
        template=template,
    input_variables=['input']
)

# Fonction pour charger la base de connaissances
def load_knowledgeBase():
    DB_FAISS_PATH = 'vectorstore/db_faiss_hopess'
    db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
    return db
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])
# Exemple d'appel pour obtenir les résultats de la recherche de similarité




def Talker(query:str, history):
    #similar_embeddings = load_knowledgeBase().similarity_search(query)
    # Utiliser le résultat de la recherche comme retriever
    retriever = load_knowledgeBase().as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3})

    # Créer une chaîne pour intégrer LLM, prompt et StrOutputParser
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # Appel de la méthode invoke avec un dictionnaire
    output = rag_chain.invoke(query)
    # Mettre à jour l'historique des messages avec la nouvelle question et réponse 
    return output





