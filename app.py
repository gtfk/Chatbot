# Versión 3.0 - Integración Completa con Supabase (Login y Personalización)
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from supabase import create_client, Client

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Chatbot Académico Duoc UC", page_icon="🤖", layout="wide")

# --- CARGA DE CLAVES DE API ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

if not GROQ_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Una o más claves de API no están configuradas. Por favor, revísalas en los Secrets de Streamlit.")
    st.stop()

# --- INICIALIZAR EL CLIENTE DE SUPABASE ---
@st.cache_resource
def init_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase_client()

# --- CACHING DE RECURSOS DEL CHATBOT ---
@st.cache_resource
def inicializar_cadena():
    # --- 1. Cargar y Procesar el PDF ---
    loader = PyPDFLoader("reglamento.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = loader.load_and_split(text_splitter=text_splitter)

    # --- 2. Crear los Embeddings y el Ensemble Retriever ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 7
    retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.7, 0.3])

    # --- 3. Conectarse al Modelo en Groq Cloud ---
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.1
    )

    # --- 4. Crear la Cadena de Conversación ---
    # ¡Modificamos el prompt para aceptar un nombre!
    prompt_template = """
    INSTRUCCIÓN PRINCIPAL: Responde SIEMPRE en español.
    Eres un asistente experto en el reglamento académico de Duoc UC. Estás hablando con un estudiante llamado {user_name}.
    Tu objetivo es dar respuestas claras y precisas basadas ÚNICAMENTE en el contexto proporcionado.
    
    INSTRUCCIÓN ESPECIAL: Si la pregunta es general (ej. "qué debe saber un alumno nuevo"), crea un resumen que cubra: Asistencia, Calificaciones y Reprobación.
    
    CONTEXTO:
    {context}
    
    PREGUNTA DEL ESTUDIANTE:
    {input}
    
    RESPUESTA:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- INICIALIZAR ESTADO DE SESIÓN ---
if 'user' not in st.session_state:
    st.session_state.user = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- LÓGICA DE AUTENTICACIÓN (PANTALLA DE LOGIN) ---
if st.session_state.user is None:
    
    st.title("🤖 Chatbot del Reglamento Académico")
    st.subheader("Por favor, inicia sesión o regístrate para continuar")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Iniciar Sesión")
        email_login = st.text_input("Email", key="login_email")
        password_login = st.text_input("Contraseña", type="password", key="login_pass")
        if st.button("Ingresar"):
            try:
                user_session = supabase.auth.sign_in_with_password({"email": email_login, "password": password_login})
                st.session_state.user = user_session.user
                
                # Cargar el perfil del usuario desde nuestra tabla 'profiles'
                profile = supabase.table('profiles').select('full_name').eq('id', user_session.user.id).execute()
                if profile.data:
                    st.session_state.user_name = profile.data[0]['full_name']
                else:
                    st.session_state.user_name = "Usuario" # Fallback
                
                st.rerun() # Reinicia el script para mostrar la app de chat
            except Exception as e:
                st.error(f"Error al iniciar sesión: {e}")

    with col2:
        st.header("Registrarse")
        name_reg = st.text_input("Nombre Completo", key="reg_name")
        email_reg = st.text_input("Email", key="reg_email")
        password_reg = st.text_input("Contraseña (mínimo 6 caracteres)", type="password", key="reg_pass")
        
        if st.button("Registrarme"):
            if len(password_reg) < 6:
                st.error("La contraseña debe tener al menos 6 caracteres.")
            elif not name_reg:
                st.error("Por favor, ingresa tu nombre completo.")
            else:
                try:
                    # 1. Crear el usuario en Supabase Auth
                    user_session = supabase.auth.sign_up({"email": email_reg, "password": password_reg})
                    
                    if user_session.user:
                        # 2. Insertar el nombre en nuestra tabla 'profiles'
                        supabase.table('profiles').insert({
                            'id': user_session.user.id, 
                            'full_name': name_reg
                        }).execute()
                        
                        st.session_state.user = user_session.user
                        st.session_state.user_name = name_reg
                        st.success("¡Registro exitoso! Serás redirigido.")
                        st.rerun()
                    else:
                        st.error("Error en el registro. El usuario podría ya existir.")
                except Exception as e:
                    st.error(f"Error al registrar: {e}")

# --- LÓGICA PRINCIPAL DEL CHATBOT (SI ESTÁ LOGUEADO) ---
else:
    # Cargar la cadena de LangChain
    retrieval_chain = inicializar_cadena()

    # Mostrar título personalizado y botón de logout
    st.title("🤖 Chatbot del Reglamento Académico")
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.caption(f"Conectado como: {st.session_state.user_name} ({st.session_state.user.email})")
    with col2:
        if st.button("Cerrar Sesión"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.session_state.user_name = None
            st.session_state.messages = []
            st.rerun()

    # Cargar historial de chat desde Supabase (solo una vez)
    if not st.session_state.messages:
        history = supabase.table('chat_history').select('role, message').eq('user_id', st.session_state.user.id).order('created_at').execute()
        for row in history.data:
            st.session_state.messages.append({"role": row['role'], "content": row['message']})

    # Mostrar mensajes del historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Procesar nueva pregunta del usuario
    if prompt := st.chat_input("¿Qué duda tienes sobre el reglamento?"):
        
        # 1. Mostrar y guardar pregunta en el estado
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 2. Guardar pregunta en Supabase
        supabase.table('chat_history').insert({
            'user_id': st.session_state.user.id,
            'role': 'user',
            'message': prompt
        }).execute()

        # 3. Generar y mostrar respuesta del bot
        with st.chat_message("assistant"):
            with st.spinner("Pensando... 💭"):
                # ¡Personalizamos la entrada con el nombre del usuario!
                response = retrieval_chain.invoke({
                    "input": prompt,
                    "user_name": st.session_state.user_name
                })
                respuesta_bot = response["answer"]
                st.markdown(respuesta_bot)
        
        # 4. Guardar respuesta en el estado
        st.session_state.messages.append({"role": "assistant", "content": respuesta_bot})
        
        # 5. Guardar respuesta en Supabase
        supabase.table('chat_history').insert({
            'user_id': st.session_state.user.id,
            'role': 'assistant',
            'message': respuesta_bot
        }).execute()