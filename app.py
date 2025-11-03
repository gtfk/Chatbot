# Versión 3.1 - Login con Google (OAuth)
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
import time # Importamos time para la redirección

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
    # ... (Esta función es idéntica a la versión anterior) ...
    loader = PyPDFLoader("reglamento.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = loader.load_and_split(text_splitter=text_splitter)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 7
    retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.7, 0.3])
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.1)
    prompt_template = """
    INSTRUCCIÓN PRINCIPAL: Responde SIEMPRE en español.
    Eres un asistente experto en el reglamento académico de Duoc UC. Estás hablando con un estudiante llamado {user_name}.
    Tu objetivo es dar respuestas claras y precisas basadas ÚNICAMENTE en el contexto proporcionado.
    INSTRUCCIÓN ESPECIAL: Si la pregunta es general (ej. "qué debe saber un alumno nuevo"), crea un resumen que cubra: Asistencia, Calificaciones y Reprobación.
    CONTEXTO: {context}
    PREGUNTA DEL ESTUDIANTE: {input}
    RESPUESTA:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# --- MANEJO DE SESIÓN DE USUARIO ---
# Intentamos obtener la sesión del usuario. Streamlit no maneja bien los redirects de OAuth,
# así que verificamos si hay un usuario en el estado de la sesión.
# El usuario tendrá que iniciar sesión y luego volver a cargar la app.
user = None
if 'user' in st.session_state:
    user = st.session_state.user
else:
    try:
        # Intenta obtener la sesión si el usuario ya está logueado en el navegador
        session = supabase.auth.get_session()
        if session and session.user:
            user = session.user
            st.session_state.user = user
    except Exception:
        pass # No hay sesión activa

# --- LÓGICA DE AUTENTICACIÓN (PANTALLA DE LOGIN) ---
if user is None:
    
    st.title("🤖 Chatbot del Reglamento Académico")
    st.subheader("Por favor, inicia sesión con tu cuenta de Google para continuar")

    # Generamos la URL de inicio de sesión de Google
    google_auth_url = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {
            "query_params": {"access_type": "offline", "prompt": "consent"},
            # Aquí puedes añadir 'hd': 'tu-dominio-academico.cl' para filtrar por correo académico
            # "hd": "alumnos.duoc.cl" 
        }
    })
    
    # Usamos st.link_button para enviar al usuario a la página de Google
    st.link_button("Iniciar Sesión con Google", google_auth_url['url'], use_container_width=True, type="primary")
    
    st.markdown("""
    **Nota Importante:** Después de iniciar sesión en la ventana de Google, serás redirigido. 
    **Deberás volver a cargar esta página del chatbot manualmente** para que la sesión se active.
    """)

# --- LÓGICA PRINCIPAL DEL CHATBOT (SI ESTÁ LOGUEADO) ---
else:
    # Cargar la cadena de LangChain
    retrieval_chain = inicializar_cadena()

    # --- OBTENER/CREAR PERFIL DE USUARIO ---
    user_name = "Estudiante" # Valor por defecto
    user_email = user.email
    user_id = user.id

    if 'user_name' not in st.session_state:
        profile = supabase.table('profiles').select('full_name').eq('id', user_id).execute()
        if profile.data:
            st.session_state.user_name = profile.data[0]['full_name']
        else:
            # Si el perfil no existe, lo creamos con el nombre de Google
            # (El email ya está en 'auth.users')
            user_full_name = user.user_metadata.get('full_name', 'Estudiante')
            supabase.table('profiles').insert({
                'id': user_id, 
                'full_name': user_full_name
            }).execute()
            st.session_state.user_name = user_full_name
    
    user_name = st.session_state.user_name

    # --- INTERFAZ DEL CHAT ---
    st.title("🤖 Chatbot del Reglamento Académico")
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.caption(f"Conectado como: {user_name} ({user_email})")
    with col2:
        if st.button("Cerrar Sesión"):
            supabase.auth.sign_out()
            st.session_state.clear() # Limpia toda la sesión
            st.rerun()

    # Cargar historial de chat desde Supabase (solo una vez)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        history = supabase.table('chat_history').select('role, message').eq('user_id', user_id).order('created_at').execute()
        for row in history.data:
            st.session_state.messages.append({"role": row['role'], "content": row['message']})

    # Mostrar mensajes del historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Procesar nueva pregunta del usuario
    if prompt := st.chat_input("¿Qué duda tienes sobre el reglamento?"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        supabase.table('chat_history').insert({
            'user_id': user_id, 'role': 'user', 'message': prompt
        }).execute()

        with st.chat_message("assistant"):
            with st.spinner("Pensando... 💭"):
                response = retrieval_chain.invoke({
                    "input": prompt,
                    "user_name": user_name
                })
                respuesta_bot = response["answer"]
                st.markdown(respuesta_bot)
        
        st.session_state.messages.append({"role": "assistant", "content": respuesta_bot})
        
        supabase.table('chat_history').insert({
            'user_id': user_id, 'role': 'assistant', 'message': respuesta_bot
        }).execute()