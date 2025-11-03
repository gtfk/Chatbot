# Versión 6.0 - Agente Investigador (sin hardcoding)
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
from langchain.agents import AgentExecutor, create_react_agent, Tool
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
def inicializar_agente_y_cadena():
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

    # --- 4. Crear la HERRAMIENTA DE BÚSQUEDA (La única herramienta) ---
    search_prompt = ChatPromptTemplate.from_template("""
    Responde la pregunta del usuario de forma clara y concisa, basándote únicamente en el siguiente contexto. Cita el artículo si lo encuentras.
    CONTEXTO: {context}
    PREGUNTA: {input}
    RESPUESTA:
    """)
    document_chain = create_stuff_documents_chain(llm, search_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    tools = [
        Tool(
            name="BuscadorReglamento",
            # Pasamos la función 'invoke' de la cadena directamente
            func=lambda input_dict: retrieval_chain.invoke(input_dict)['answer'],
            description="""
            Es la única herramienta. Úsala para buscar y responder CUALQUIER pregunta sobre el reglamento académico, 
            como asistencia, notas, reprobación, artículos, etc. 
            La entrada debe ser una pregunta clara.
            """
        ),
    ]

    # --- 5. Crear el CEREBRO DEL AGENTE (Prompt) ---
    template = """
    INSTRUCCIÓN PRINCIPAL: Responde SIEMPRE en español, con un tono amigable y cercano.
    
    PERSONAJE: Eres un asistente experto en el reglamento académico de Duoc UC. Estás hablando con un estudiante llamado {user_name}.
    
    REGLAS DE RAZONAMIENTO:
    1.  **Si la pregunta es específica** (ej. "porcentaje de asistencia", "artículo 20"), usa la herramienta "BuscadorReglamento" una sola vez y da la respuesta.
    2.  **Si la pregunta es general** (ej. "qué debo saber como alumno nuevo", "resumen del reglamento"), TU TRABAJO es descomponerla. Debes usar la herramienta "BuscadorReglamento" MÚLTIPLES VECES para encontrar los datos clave. Busca al menos:
        - El porcentaje de asistencia (usa una consulta como "cuál es el porcentaje de asistencia").
        - La nota mínima para aprobar (usa una consulta como "cuál es la nota mínima para aprobar").
        - Las causas de reprobación (usa una consulta como "cuáles son las causas de reprobación").
    3.  **Después de buscar** todos los datos, sintetiza (resume) la información que encontraste en una respuesta amigable y completa para {user_name}.
    4.  Dirígete a {user_name} por su nombre al menos una vez en la respuesta.

    HERRAMIENTAS DISPONIBLES:
    {tools}

    Usa el siguiente formato:

    Pregunta: la pregunta original que debes responder
    Pensamiento: siempre debes pensar qué hacer a continuación
    Acción: la acción a tomar, debe ser una de [{tool_names}]
    Entrada de la Acción: la consulta de búsqueda para la herramienta (debe ser una pregunta)
    Observación: el resultado de la acción
    ... (este patrón de Pensamiento/Acción/Entrada de la Acción/Observación puede repetirse N veces)
    Pensamiento: Ahora sé la respuesta final.
    Respuesta Final: la respuesta final a la pregunta original del usuario.

    ¡Comienza!

    Pregunta: {input}
    Pensamiento:{agent_scratchpad}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor

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
    
    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
    with col1:
        st.caption(f"Conectado como: {user_name} ({user_email})")
    
    with col2:
        if st.button("Limpiar Chat", use_container_width=True):
            try:
                supabase.table('chat_history').delete().eq('user_id', st.session_state.user_id).execute()
                st.session_state.messages = []
                welcome_message = f"¡Hola {user_name}! Tu historial ha sido limpiado. ¿En qué te puedo ayudar?"
                st.session_state.messages.append({"role": "assistant", "content": welcome_message})
                supabase.table('chat_history').insert({
                    'user_id': st.session_state.user_id, 'role': 'assistant', 'message': welcome_message
                }).execute()
                st.rerun() 
            except Exception as e:
                st.error(f"No se pudo limpiar el historial: {e}")

    with col3:
        authenticator.logout(button_name='Cerrar Sesión', location='main', key='logout_button')
    
    # Inicializamos el Agente
    agent_executor = inicializar_agente_y_cadena()

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
                # ¡Ahora invocamos al Agente!
                response = agent_executor.invoke({
                    "input": prompt,
                    "user_name": user_name 
                })
                respuesta_bot = response["output"] # El Agente devuelve 'output'
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

    st.markdown("---")
    st.subheader("¿Olvidaste tu contraseña?")
    st.markdown("""
    Debido a que este es un sistema auto-gestionado, no hay un reseteo de contraseña automático.
    
    **Solución:** Pídele al administrador del proyecto que **borre tu usuario**
    desde la tabla `profiles` en **Supabase**. Una vez borrado, podrás registrarte de nuevo con el mismo email.
    """)

    # --- FORMULARIO DE REGISTRO PERSONALIZADO (en la barra lateral) ---
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