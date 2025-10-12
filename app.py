import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

st.set_page_config(page_title="Chatbot Académico Duoc UC", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot del Reglamento Académico")

HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    st.error("La clave de API de Hugging Face no está configurada. Por favor, agrégala a las variables de entorno.")
    st.stop()

@st.cache_resource
def cargar_retriever_y_cadena():
    nombre_del_archivo = "RES-VRA-03-2024-NUEVO-REGLAMENTO-ACADÉMICO63-1.pdf"
    loader = PyPDFLoader(nombre_del_archivo)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5
    
    retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.8, 0.2])

    endpoint = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        temperature=0.1,
        max_new_tokens=1024,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    llm = ChatHuggingFace(llm=endpoint)

    prompt = ChatPromptTemplate.from_template("""
    INSTRUCCIONES:
    - Eres un asistente académico experto. Responde la pregunta del usuario de forma clara y directa.
    - Basa tu respuesta ÚNICAMENTE en el contexto proporcionado.
    - Si la respuesta no está en el contexto, di "No encuentro información sobre eso en el reglamento."

    CONTEXTO:
    {context}

    PREGUNTA DEL USUARIO:
    {input}

    RESPUESTA DIRECTA:
    """)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

try:
    retrieval_chain = cargar_retriever_y_cadena()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("¿Qué duda tienes sobre el reglamento?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando... 💭"):
                response = retrieval_chain.invoke({"input": prompt})
                st.markdown(response["answer"])
        
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

except Exception as e:
    st.error(f"Ha ocurrido un error: {e}")