import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURACIÓN BÁSICA ---
st.set_page_config(page_title="DuocBot v1.0", page_icon="🤖")

st.title("🤖 Chatbot Reglamento Académico")
st.markdown("### Versión Beta 1.0")
st.info("Bienvenido. Este es un prototipo experimental para consultas del reglamento.")

# --- CARGA DE CLAVES (Usa tus secrets actuales) ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ Error: No se detectó la API Key de Groq.")
    st.stop()

# --- MOTOR DEL CHATBOT (Simplificado) ---
@st.cache_resource
def iniciar_motor():
    # Asumimos que el archivo 'reglamento.pdf' está en la carpeta
    if not os.path.exists("reglamento.pdf"):
        st.error("No se encuentra el archivo reglamento.pdf")
        return None

    loader = PyPDFLoader("reglamento.pdf")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0)
    
    system_prompt = (
        "Eres un asistente de la institución Duoc UC. "
        "Responde las preguntas basándote solo en el contexto proporcionado. "
        "Si no sabes la respuesta, di que no lo sabes."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

# --- INTERFAZ DE CHAT (Sin Base de Datos, Memoria Volátil) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Inicializar cadena
chain = iniciar_motor()

# Mostrar historial (Solo de la sesión actual)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input de usuario
if prompt := st.chat_input("Escribe tu pregunta sobre el reglamento..."):
    # 1. Mostrar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generar respuesta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        if chain:
            with st.spinner("Analizando documento PDF..."):
                response = chain.invoke({"input": prompt})
                full_response = response["answer"]
                message_placeholder.markdown(full_response)
        
            st.session_state.messages.append({"role": "assistant", "content": full_response})