import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
st.set_page_config(page_title="Chatbot Académico Duoc UC", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot del Reglamento Académico")

# --- CACHING DE RECURSOS ---
# Usamos el caché de Streamlit para no tener que cargar el modelo y los documentos cada vez.
@st.cache_resource
def cargar_modelo_y_retriever():
    # --- 1. Cargar y Procesar el PDF ---
    nombre_del_archivo = "RES-VRA-03-2024-NUEVO-REGLAMENTO-ACADÉMICO63-1.pdf"
    loader = PyPDFLoader(nombre_del_archivo)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "Artículo", ".", ","]
    )
    docs = text_splitter.split_documents(pages)

    # --- 2. Crear el Ensemble Retriever (el "equipo de buscadores") ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # (Asegúrate de que rank_bm25 esté en requirements.txt)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5
    
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.8, 0.2]
    )

    # --- 3. Cargar el Modelo de Lenguaje Localmente ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_id = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # --- 4. Crear la Cadena de Conversación ---
    prompt = ChatPromptTemplate.from_template("""
    INSTRUCCIONES:
    - Eres un asistente académico experto en el reglamento de Duoc UC.
    - Tu tarea es responder la pregunta del usuario de forma clara y directa.
    - Basa tu respuesta ÚNICAMENTE en el siguiente contexto. No inventes información.
    - Si la respuesta está en el contexto, extráela y preséntala de forma concisa.
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

# --- LÓGICA DE LA APLICACIÓN DE CHAT ---
try:
    retrieval_chain = cargar_modelo_y_retriever()

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
    st.error(f"Ha ocurrido un error al cargar el modelo o procesar el documento: {e}")
    st.info("Esto puede deberse a limitaciones de memoria en el entorno de despliegue. El modelo Zephyr-7B requiere una cantidad significativa de RAM.")