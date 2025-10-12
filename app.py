import streamlit as st
<<<<<<< HEAD
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
=======
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
>>>>>>> b40369bc9d258bbb3d32963b65824b0b558fb4e6
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

<<<<<<< HEAD
# --- CONFIGURACIÓN DE LA PÁGINA Y API KEY ---
st.set_page_config(page_title="Chatbot Académico Duoc UC", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot del Reglamento Académico")

# Cargar la API Key de Hugging Face desde los secrets de Streamlit/Vercel
# ¡Asegúrate de haber configurado esta variable de entorno en Vercel!
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    st.error("La clave de API de Hugging Face no está configurada. Por favor, agrégala a las variables de entorno de Vercel.")
    st.stop()

# --- CACHING DE RECURSOS ---
@st.cache_resource
def cargar_retriever_y_cadena():
=======
# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
st.set_page_config(page_title="Chatbot Académico Duoc UC", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot del Reglamento Académico")

# --- CACHING DE RECURSOS ---
# Usamos el caché de Streamlit para no tener que cargar el modelo y los documentos cada vez.
@st.cache_resource
def cargar_modelo_y_retriever():
>>>>>>> b40369bc9d258bbb3d32963b65824b0b558fb4e6
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

<<<<<<< HEAD
    # --- 2. Crear el Ensemble Retriever ---
=======
    # --- 2. Crear el Ensemble Retriever (el "equipo de buscadores") ---
>>>>>>> b40369bc9d258bbb3d32963b65824b0b558fb4e6
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
<<<<<<< HEAD
=======
    # (Asegúrate de que rank_bm25 esté en requirements.txt)
>>>>>>> b40369bc9d258bbb3d32963b65824b0b558fb4e6
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5
    
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.8, 0.2]
    )

<<<<<<< HEAD
    # --- 3. Conectarse al Modelo de Lenguaje vía API ---
    endpoint = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        temperature=0.1,
        max_new_tokens=1024,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    llm = ChatHuggingFace(llm=endpoint)
=======
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
>>>>>>> b40369bc9d258bbb3d32963b65824b0b558fb4e6

    # --- 4. Crear la Cadena de Conversación ---
    prompt = ChatPromptTemplate.from_template("""
    INSTRUCCIONES:
    - Eres un asistente académico experto en el reglamento de Duoc UC.
    - Tu tarea es responder la pregunta del usuario de forma clara y directa.
    - Basa tu respuesta ÚNICAMENTE en el siguiente contexto. No inventes información.
<<<<<<< HEAD
=======
    - Si la respuesta está en el contexto, extráela y preséntala de forma concisa.
>>>>>>> b40369bc9d258bbb3d32963b65824b0b558fb4e6
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
<<<<<<< HEAD
    retrieval_chain = cargar_retriever_y_cadena()
=======
    retrieval_chain = cargar_modelo_y_retriever()
>>>>>>> b40369bc9d258bbb3d32963b65824b0b558fb4e6

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
<<<<<<< HEAD
    st.error(f"Ha ocurrido un error: {e}")
=======
    st.error(f"Ha ocurrido un error al cargar el modelo o procesar el documento: {e}")
    st.info("Esto puede deberse a limitaciones de memoria en el entorno de despliegue. El modelo Zephyr-7B requiere una cantidad significativa de RAM.")
>>>>>>> b40369bc9d258bbb3d32963b65824b0b558fb4e6
