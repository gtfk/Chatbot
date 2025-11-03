# Versión 5.2 - Equilibrio final: Conciso Y Personalizado
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
import streamlit_authenticator as stauth
import time

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
    # ... (Carga de PDF y Retrievers - sin cambios) ...
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

    # --- CAMBIO CLAVE: INSTRUCCIÓN EXPLÍCITA DE USAR EL NOMBRE ---
    prompt_template = """
    INSTRUCCIÓN PRINCIPAL: Responde SIEMPRE en español, con un tono amigable y cercano.
    
    PERSONAJE: Eres un asistente experto en el reglamento académico de Duoc UC. Estás hablando con un estudiante llamado {user_name}.
    
    REGLAS IMPORTANTES:
    1. Dirígete a {user_name} por su nombre al menos una vez en la respuesta (ej. "Claro, {user_name}, te explico...").
    2. Da una respuesta clara, concisa y directa.
    3. Basa tu respuesta ÚNICAMENTE en el contexto proporcionado.
    4. Cita el artículo (ej. "Artículo N°30") si lo encuentras.
    5. NO añadas información extra que no fue solicitada.

    INSTRUCCIÓN ESPECIAL: Si la pregunta es general (ej. "qué debe saber un alumno nuevo"), crea un resumen que cubra: Asistencia, Calificaciones y Reprobación.

    CONTEXTO:
    {context}

    PREGUNTA DE {user_name}:
    {input}

    RESPUESTA:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    # --- FIN DEL CAMBIO ---
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# --- LÓGICA DE AUTENTICACIÓN ---

# 1. Cargar todos los perfiles de usuario desde Supabase
def fetch_all_users():
    try:
        response = supabase.table('profiles').select("email, full_name, password_hash").execute()
        users = response.data
        if not users:
            return {'usernames': {}}
        credentials = {'usernames': {}}
        for user in users:
            credentials['usernames'][user['email']] = {
                'email': user['email'],
                'name': user['full_name'],
                'password': user['password_hash']
            }
        return credentials
    except Exception as e:
        st.error(f"Error al cargar usuarios: {e}")
        return {'usernames': {}}

# 2. Configurar el Autenticador
credentials = fetch_all_users()
authenticator = stauth.Authenticate(
    credentials,
    'chatbot_duoc_cookie',
    'abcdefg123456', # ¡Recuerda cambiar esto!
    cookie_expiry_days=30
)

# --- LÓGICA DE LA APLICACIÓN ---

st.title("🤖 Chatbot del Reglamento Académico")

# 3. Comprobar si el usuario ya está logueado
if st.session_state["authentication_status"] is True:
    # --- Si el login es exitoso ---
    user_name = st.session_state["name"]
    user_email = st.session_state["username"]
    
    authenticator.logout('Cerrar Sesión')
    st.caption(f"Conectado como: {user_name} ({user_email})")
    
    retrieval_chain = inicializar_cadena()

    # Cargar historial de chat desde Supabase
    if 'user_id' not in st.session_state:
        user_id_response = supabase.table('profiles').select('id').eq('email', user_email).execute()
        if user_id_response.data:
            st.session_state.user_id = user_id_response.data[0]['id']
        else:
            st.error("Error crítico: No se pudo encontrar el perfil del usuario logueado.")
            st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        history = supabase.table('chat_history').select('role, message').eq('user_id', st.session_state.user_id).order('created_at').execute()
        for row in history.data:
            st.session_state.messages.append({"role": row['role'], "content": row['message']})
        
        # Saludo de bienvenida si el historial está vacío
        if not st.session_state.messages:
            welcome_message = f"¡Hola {user_name}! Soy tu asistente del reglamento académico. ¿En qué te puedo ayudar hoy?"
            st.session_state.messages.append({"role": "assistant", "content": welcome_message})
            supabase.table('chat_history').insert({
                'user_id': st.session_state.user_id, 'role': 'assistant', 'message': welcome_message
            }).execute()

    # Mostrar mensajes del historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Procesar nueva pregunta
    if prompt := st.chat_input("¿Qué duda tienes sobre el reglamento?"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        supabase.table('chat_history').insert({
            'user_id': st.session_state.user_id, 'role': 'user', 'message': prompt
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
            'user_id': st.session_state.user_id, 'role': 'assistant', 'message': respuesta_bot
        }).execute()

# 4. Si el usuario NO está logueado, mostrar Login y Registro
else:
    authenticator.login(location='main')
    
    if st.session_state["authentication_status"] is False:
        st.error('Email o contraseña incorrecta')
    elif st.session_state["authentication_status"] is None:
        st.info('Por favor, ingresa tu email y contraseña. ¿Nuevo usuario? Registrate en la barra lateral.')

    with st.sidebar:
        st.subheader("¿Nuevo Usuario? Regístrate")
        with st.form(key="register_form", clear_on_submit=True):
            name_reg = st.text_input("Nombre Completo")
            email_reg = st.text_input("Email")
            password_reg = st.text_input("Contraseña", type="password")
            confirm_password_reg = st.text_input("Confirmar Contraseña", type="password")
            submit_button = st.form_submit_button(label="Registrarse")

            if submit_button:
                if not name_reg: st.error("Por favor, ingresa tu nombre.")
                elif not email_reg: st.error("Por favor, ingresa tu email.")
                elif password_reg != confirm_password_reg: st.error("Las contraseñas no coinciden.")
                elif len(password_reg) < 6: st.error("La contraseña debe tener al menos 6 caracteres.")
                else:
                    try:
                        hasher = stauth.Hasher()
                        hashed_password = hasher.hash(password_reg)
                        
                        insert_response = supabase.table('profiles').insert({
                            'full_name': name_reg,
                            'email': email_reg,
                            'password_hash': hashed_password
                        }).execute()
                        
                        if insert_response.data:
                            st.success('¡Usuario registrado! Ahora puedes iniciar sesión.')
                            time.sleep(2) 
                        else:
                            st.error('Error al registrar el usuario en la base de datos.')
                    
                    except Exception as e:
                        if 'duplicate key value violates unique constraint' in str(e):
                            st.error("Error: Ese email ya está registrado.")
                        else:
                            st.error(f"Error en el registro: {e}")