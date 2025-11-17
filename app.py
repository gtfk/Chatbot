# Versión 7.0 - Agente de LangChain (RAG + Inscripción)
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
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# Importaciones clave para el Agente
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import os
from supabase import create_client, Client
import streamlit_authenticator as stauth
import time
from datetime import time as dt_time # Para comparar horarios

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Académico Duoc UC", page_icon="🤖", layout="wide")

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
# Esta función ahora solo prepara los componentes caros (LLM y Retriever)
@st.cache_resource
def inicializar_componentes():
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
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.1)
    
    return llm, retriever

# --- LÓGICA DE AUTENTICACIÓN ---
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

credentials = fetch_all_users()
authenticator = stauth.Authenticate(
    credentials,
    'chatbot_duoc_cookie',
    'abcdefg123456', # ¡Recuerda cambiar esto!
    cookie_expiry_days=30
)

# --- INICIO DE LA LÓGICA DE LA APLICACIÓN ---
st.title("🤖 Asistente Académico Duoc UC")

# 3. Comprobar si el usuario ya está logueado
if st.session_state["authentication_status"] is True:
    user_name = st.session_state["name"]
    user_email = st.session_state["username"]
    
    # Cargar user_id en la sesión
    if 'user_id' not in st.session_state:
        user_id_response = supabase.table('profiles').select('id').eq('email', user_email).execute()
        if user_id_response.data:
            st.session_state.user_id = user_id_response.data[0]['id']
        else:
            st.error("Error crítico: No se pudo encontrar el perfil del usuario logueado.")
            st.stop()
    
    user_id = st.session_state.user_id

    # --- Encabezado y Logout ---
    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
    with col1:
        st.caption(f"Conectado como: {user_name} ({user_email})")
    with col2:
        if st.button("Limpiar Chat", use_container_width=True, key="clear_chat"):
            supabase.table('chat_history').delete().eq('user_id', user_id).execute()
            st.session_state.messages = []
            welcome_message = f"¡Hola {user_name}! Tu historial ha sido limpiado. ¿En qué te puedo ayudar hoy?"
            st.session_state.messages.append({"role": "assistant", "content": welcome_message})
            supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': welcome_message}).execute()
            st.rerun() 
    with col3:
        if st.button("Cerrar Sesión", use_container_width=True, key="logout_button_global"):
            authenticator.logout()
            st.session_state.clear()
            st.rerun()
            
    st.divider()

    # --- INICIALIZACIÓN DEL AGENTE Y HERRAMIENTAS ---
    # Cargamos los componentes caros (LLM y Retriever) desde el caché
    llm, retriever = inicializar_componentes()

    # --- Definición de Herramientas ---
    
    # Herramienta 1: Buscador de Reglamento (RAG)
    rag_prompt_template = """
    INSTRUCCIÓN PRINCIPAL: Responde SIEMPRE en español, con un tono amigable y cercano.
    PERSONAJE: Eres un asistente experto en el reglamento académico de Duoc UC. Estás hablando con un estudiante llamado {user_name}.
    REGLAS IMPORTANTES:
    1. Dirígete a {user_name} por su nombre al menos una vez en la respuesta.
    2. Da una respuesta clara, concisa y directa.
    3. Basa tu respuesta ÚNICAMENTE en el contexto proporcionado.
    4. Cita el artículo (ej. "Artículo N°30") si lo encuentras.

    INSTRUCCIÓN ESPECIAL: 
    Si la pregunta del usuario es general sobre ser un "alumno nuevo" o "qué debería saber", 
    IGNORA EL CONTEXTO y responde EXACTAMENTE con este resumen:
    "¡Hola {user_name}! Como alumno nuevo, lo más importante que debes saber del reglamento es:
    
    1.  **Asistencia (Art. 30):** Debes cumplir con un **70% de asistencia** tanto en las actividades teóricas como en las prácticas para aprobar.
    2.  **Calificaciones (Art. 37):** La nota mínima para aprobar una asignatura es un **4,0**.
    3.  **Reprobación (Art. 39):** Repruebas una asignatura si tu nota final es inferior a 4,0 o si no cumples con el 70% de asistencia.
    
    ¡Espero que esto te ayude, {user_name}! Si tienes otra duda más específica, solo pregunta."

    Si la pregunta NO es general, sigue las reglas normales y usa el contexto.

    CONTEXTO:
    {context}
    PREGUNTA DE {user_name}:
    {input}
    RESPUESTA:
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Envolvemos la cadena en una función simple para la herramienta
    def run_rag_chain(query: str) -> str:
        """Responde preguntas sobre el reglamento académico."""
        response = retrieval_chain.invoke({"input": query, "user_name": user_name})
        return response["answer"]

    # Herramienta 2: Buscador de Asignaturas
    def get_user_schedule(user_uuid):
        user_regs = supabase.table('registrations').select('section_id').eq('user_id', user_uuid).execute().data
        if not user_regs: return [], []
        section_ids = [reg['section_id'] for reg in user_regs]
        schedule_data = supabase.table('sections').select('subject_id, day_of_week, start_time, end_time').in_('id', section_ids).execute().data
        schedule = []
        registered_subject_ids = []
        for sec in schedule_data:
            schedule.append({
                "day": sec['day_of_week'],
                "start": dt_time.fromisoformat(sec['start_time']),
                "end": dt_time.fromisoformat(sec['end_time'])
            })
            registered_subject_ids.append(sec['subject_id'])
        return schedule, registered_subject_ids

    def buscar_asignaturas(nombre_asignatura: str) -> str:
        """Busca secciones disponibles para una asignatura, verificando cupos y si el usuario ya la inscribió."""
        try:
            # 1. Buscar el ID de la asignatura
            subject_response = supabase.table('subjects').select('id').ilike('name', f'%{nombre_asignatura}%').execute()
            if not subject_response.data:
                return f"Lo siento {user_name}, no encontré ninguna asignatura con el nombre '{nombre_asignatura}'."
            
            selected_subject_id = subject_response.data[0]['id']
            
            # 2. Verificar si el usuario ya la inscribió
            _ , registered_subject_ids = get_user_schedule(user_id)
            if selected_subject_id in registered_subject_ids:
                return f"{user_name}, ya tienes esa asignatura inscrita en otra sección. Debes anularla primero si quieres cambiarla."

            # 3. Buscar secciones disponibles
            sections_response = supabase.table('sections').select('*').eq('subject_id', selected_subject_id).execute()
            if not sections_response.data:
                return f"No hay secciones disponibles para '{nombre_asignatura}' en este momento."

            # 4. Formatear la respuesta
            respuesta = f"¡Claro, {user_name}! Encontré estas secciones para '{nombre_asignatura}':\n"
            secciones_encontradas = 0
            
            for sec in sections_response.data:
                registrations_count_response = supabase.table('registrations').select('id', count='exact').eq('section_id', sec['id']).execute()
                registrations_count = registrations_count_response.count
                cupos_disponibles = sec['capacity'] - (registrations_count if registrations_count else 0)
                
                if cupos_disponibles > 0:
                    secciones_encontradas += 1
                    respuesta += f"\n* **Sección {sec['section_code']}**: {sec['day_of_week']} de {sec['start_time']} a {sec['end_time']}. Quedan {cupos_disponibles} cupos. (Prof: {sec['professor_name']})"
            
            if secciones_encontradas == 0:
                return f"Lo siento {user_name}, todas las secciones para '{nombre_asignatura}' están llenas."
            
            respuesta += f"\n\nPara inscribirte, dime: 'inscribe la sección [código]', por ejemplo: 'inscribe la sección {sections_response.data[0]['section_code']}'."
            return respuesta
            
        except Exception as e:
            return f"Error al buscar asignaturas: {e}"

    # Herramienta 3: Inscriptor de Asignaturas
    class InscribirInput(BaseModel):
        section_code: str = Field(description="El código de la sección a inscribir, por ejemplo '001D' o '002D'")

    def inscribir_asignatura(section_code: str) -> str:
        """Inscribe al usuario en una sección específica, validando cupos y tope de horario."""
        try:
            # 1. Encontrar la sección
            section_response = supabase.table('sections').select('*').eq('section_code', section_code.upper()).execute()
            if not section_response.data:
                return f"No pude encontrar la sección '{section_code}'. Por favor, verifica el código."
            
            section_to_register = section_response.data[0]
            
            # 2. Verificar si la asignatura ya está inscrita
            user_schedule, registered_subject_ids = get_user_schedule(user_id)
            if section_to_register['subject_id'] in registered_subject_ids:
                return f"Error: Ya tienes esta asignatura inscrita en otra sección. Debes anularla primero."
            
            # 3. Verificar cupos
            registrations_count_response = supabase.table('registrations').select('id', count='exact').eq('section_id', section_to_register['id']).execute()
            registrations_count = registrations_count_response.count
            cupos_disponibles = section_to_register['capacity'] - (registrations_count if registrations_count else 0)
            
            if cupos_disponibles <= 0:
                return f"Lo siento, {user_name}, la sección {section_code} se acaba de llenar. No quedan cupos."
            
            # 4. Verificar tope de horario
            def check_schedule_conflict(user_schedule, new_section):
                new_day = new_section['day_of_week']
                new_start = dt_time.fromisoformat(new_section['start_time'])
                new_end = dt_time.fromisoformat(new_section['end_time'])
                for scheduled in user_schedule:
                    if scheduled['day'] == new_day:
                        if max(scheduled['start'], new_start) < min(scheduled['end'], new_end):
                            return True 
                return False
            
            if check_schedule_conflict(user_schedule, section_to_register):
                return f"¡Error! La sección {section_code} ({section_to_register['day_of_week']} {section_to_register['start_time']}) tiene un tope de horario con otra asignatura que ya tienes."

            # 5. Inscribir
            supabase.table('registrations').insert({
                'user_id': user_id,
                'section_id': section_to_register['id']
            }).execute()
            
            return f"¡Éxito, {user_name}! Has sido inscrito en la sección {section_code}."
            
        except Exception as e:
            return f"Error al inscribir: {e}"

    # --- Creación del Agente ---
    tools = [
        Tool(
            name="BuscadorDeReglamento",
            func=run_rag_chain,
            description="Útil para responder preguntas específicas sobre el reglamento académico, artículos, normas, asistencia, y preguntas generales de 'alumno nuevo'."
        ),
        Tool(
            name="BuscadorDeAsignaturas",
            func=buscar_asignaturas,
            description="Útil para buscar qué secciones, horarios y cupos hay disponibles para una asignatura. La entrada es el nombre de la asignatura (ej. 'Matemáticas I')."
        ),
        Tool(
            name="InscriptorDeAsignaturas",
            func=inscribir_asignatura,
            description="Útil para inscribir al usuario en una sección de una asignatura. La entrada es el código de la sección (ej. '001D').",
            args_schema=InscribirInput
        ),
    ]

    # Prompt del Agente (en español)
    agent_prompt_template = """
    INSTRUCCIÓN PRINCIPAL: Responde SIEMPRE en español, con un tono amigable y cercano.
    
    PERSONAJE: Eres un asistente académico de Duoc UC. Estás hablando con un estudiante llamado {user_name}.
    
    REGLAS DE RAZONAMIENTO:
    1.  Tu trabajo es entender la intención del usuario y usar la herramienta correcta.
    2.  Si la pregunta es sobre el reglamento (asistencia, notas, artículos, alumno nuevo), usa "BuscadorDeReglamento".
    3.  Si el usuario quiere SABER sobre asignaturas (ej. "qué secciones hay de cálculo"), usa "BuscadorDeAsignaturas".
    4.  Si el usuario quiere INSCRIBIR una sección (ej. "inscríbeme en la 001D"), usa "InscriptorDeAsignaturas".
    5.  Dirígete a {user_name} por su nombre.

    HERRAMIENTAS DISPONIBLES:
    {tools}

    Usa el siguiente formato:

    Pregunta: la pregunta original que debes responder
    Pensamiento: siempre debes pensar qué hacer a continuación
    Acción: la acción a tomar, debe ser una de [{tool_names}]
    Entrada de la Acción: la entrada para la acción (si es InscriptorDeAsignaturas, debe ser solo el código de sección)
    Observación: el resultado de la acción
    ... (este patrón de Pensamiento/Acción/Entrada de la Acción/Observación puede repetirse N veces)
    Pensamiento: Ahora sé la respuesta final.
    Respuesta Final: la respuesta final a la pregunta original del usuario.

    ¡Comienza!

    Pregunta: {input}
    Pensamiento:{agent_scratchpad}
    """
    
    agent_prompt = PromptTemplate.from_template(agent_prompt_template)
    
    agent = create_react_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # --- FIN DE LA INICIALIZACIÓN DEL AGENTE ---
    
    # Cargar historial de chat desde Supabase
    if "messages" not in st.session_state:
        st.session_state.messages = []
        history = supabase.table('chat_history').select('role, message').eq('user_id', user_id).order('created_at').execute()
        for row in history.data:
            st.session_state.messages.append({"role": row['role'], "content": row['message']})
        
        # Saludo de bienvenida si el historial está vacío
        if not st.session_state.messages:
            welcome_message = f"¡Hola {user_name}! Soy tu asistente académico. Puedes preguntarme sobre el reglamento o pedirme que busque o inscriba asignaturas."
            st.session_state.messages.append({"role": "assistant", "content": welcome_message})
            supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': welcome_message}).execute()

    # Mostrar mensajes del historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Procesar nueva pregunta
    if prompt := st.chat_input("¿Qué duda tienes?"):
        
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