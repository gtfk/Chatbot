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
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain import hub # Necesario para cargar el prompt oficial
import os

# --- CONFIGURACIÓN DE LA PÁGINA Y API KEY ---
st.set_page_config(page_title="Chatbot Académico Duoc UC", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot del Reglamento Académico")

HUGGINGFACEHUB_API_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    st.error("La clave de API de Hugging Face no está configurada. Por favor, agrégala a los Secrets de Streamlit.")
    st.stop()

# --- CACHING DE RECURSOS ---
@st.cache_resource
def inicializar_agente():
    # --- 1. Cargar y Procesar el PDF ---
    nombre_del_archivo = "reglamento.pdf"
    loader = PyPDFLoader(nombre_del_archivo)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)

    # --- 2. Crear el Ensemble Retriever ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5
    retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.8, 0.2])

    # --- 3. Conectarse al Modelo de Lenguaje vía API ---
    endpoint = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation", temperature=0.1, max_new_tokens=1024,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    llm = ChatHuggingFace(llm=endpoint)

    # --- 4. Crear la HERRAMIENTA N°1: Buscador de Reglamento ---
    search_prompt = ChatPromptTemplate.from_template("Responde la pregunta del usuario de forma clara y concisa, basándote únicamente en el siguiente contexto. Cita el artículo si lo encuentras.\n\nCONTEXTO: {context}\n\nPREGUNTA: {input}\n\nRESPUESTA:")
    document_chain = create_stuff_documents_chain(llm, search_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # --- 5. Crear la HERRAMIENTA N°2: Guía para Alumnos Nuevos ---
    def guia_alumno_nuevo(query: str) -> str:
        return """
        ¡Hola y bienvenido a Duoc UC! Como alumno nuevo, aquí tienes un resumen de los puntos más importantes del reglamento:
        
        * **Asistencia:** Es un requisito clave. Según el Artículo N°30, debes cumplir con un **70% de asistencia** tanto en las actividades teóricas como prácticas para poder aprobar una asignatura.
        * **Calificaciones:** Para aprobar una asignatura, necesitas una nota final igual o superior a **4,0** (Artículo N°37).
        * **Reprobación:** Repruebas una asignatura si tu nota final es inferior a 4,0 o si no cumples con el porcentaje mínimo de asistencia (Artículo N°39).
        * **Conducta:** Se espera que mantengas una conducta de respeto y sana convivencia. Actos como copiar en exámenes o la agresión física/verbal son considerados faltas graves y pueden llevar a sanciones (Título XXI).

        ¡Mucho éxito en tu primer semestre! Si tienes una pregunta más específica, no dudes en consultarme.
        """

    # --- 6. Definir las Herramientas para el Agente ---
    tools = [
        Tool(
            name="Busqueda Reglamento Especifico",
            func=retrieval_chain.invoke,
            description="Útil para responder preguntas específicas y concretas sobre artículos, notas, asistencia, fechas o procedimientos del reglamento académico."
        ),
        Tool(
            name="Resumen Alumno Nuevo",
            func=guia_alumno_nuevo,
            description="Útil para responder preguntas generales sobre qué necesita saber un alumno nuevo, consejos para empezar o información importante para el primer día."
        ),
    ]

    # --- 7. Crear el Agente (CON EL PROMPT CORREGIDO) ---
    # Obtenemos la plantilla oficial de LangChain que sí incluye las variables {tools} y {tool_names}.
    prompt = hub.pull("hwchase17/react")
    # Añadimos nuestra instrucción de idioma al final.
    prompt.template = "Responde siempre en español.\n\n" + prompt.template
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor

# --- LÓGICA DE LA APLICACIÓN DE CHAT ---
try:
    agent_executor = inicializar_agente()

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
                response = agent_executor.invoke({"input": prompt})
                st.markdown(response["output"])
        
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})

except Exception as e:
    st.error(f"Ha ocurrido un error: {e}")