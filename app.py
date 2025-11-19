# Versión 8.0 - Chatbot con Inscripción Integrada
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
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
import os
from supabase import create_client, Client
import streamlit_authenticator as stauth
import time
from datetime import time as dt_time
import json

# --- URLs DE LOGOS ---
LOGO_BANNER_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Logo_DuocUC.svg/2560px-Logo_DuocUC.svg.png"
LOGO_ICON_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlve2kMlU53cq9Tl0DMxP0Ffo0JNap2dXq4q_uSdf4PyFZ9uraw7MU5irI6mA-HG8byNI&usqp=CAU"

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Chatbot Académico Duoc UC", 
    page_icon=LOGO_ICON_URL,
    layout="wide"
)

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

# --- FUNCIONES AUXILIARES PARA INSCRIPCIÓN ---
def get_user_schedule_data(user_uuid):
    """Obtiene el horario actual del usuario"""
    user_regs = supabase.table('registrations').select('section_id').eq('user_id', user_uuid).execute().data
    if not user_regs: 
        return [], []
    
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

def check_schedule_conflict_func(user_schedule, new_section):
    """Verifica si hay conflicto de horario"""
    new_day = new_section['day_of_week']
    new_start = dt_time.fromisoformat(new_section['start_time'])
    new_end = dt_time.fromisoformat(new_section['end_time'])
    
    for scheduled in user_schedule:
        if scheduled['day'] == new_day:
            if max(scheduled['start'], new_start) < min(scheduled['end'], new_end):
                return True 
    return False

# --- HERRAMIENTAS DE LANGCHAIN PARA EL CHATBOT ---
@tool
def buscar_asignaturas(query: str = "") -> str:
    """
    Busca asignaturas disponibles. Si no se proporciona query, devuelve todas las asignaturas.
    
    Args:
        query: Nombre o parte del nombre de la asignatura a buscar (opcional)
    
    Returns:
        Lista de asignaturas encontradas en formato legible
    """
    try:
        if query:
            subjects_response = supabase.table('subjects').select('id, name').ilike('name', f'%{query}%').order('name').execute()
        else:
            subjects_response = supabase.table('subjects').select('id, name').order('name').execute()
        
        subjects = subjects_response.data
        
        if not subjects:
            return "No se encontraron asignaturas."
        
        result = "📚 **Asignaturas encontradas:**\n\n"
        for i, subj in enumerate(subjects, 1):
            result += f"{i}. {subj['name']} (ID: {subj['id']})\n"
        
        return result
    except Exception as e:
        return f"Error al buscar asignaturas: {str(e)}"

@tool
def ver_secciones_asignatura(subject_name: str) -> str:
    """
    Muestra las secciones disponibles de una asignatura específica.
    
    Args:
        subject_name: Nombre exacto de la asignatura
    
    Returns:
        Información detallada de las secciones disponibles
    """
    try:
        # Buscar la asignatura
        subject_response = supabase.table('subjects').select('id, name').eq('name', subject_name).execute()
        
        if not subject_response.data:
            return f"No se encontró la asignatura '{subject_name}'. Usa buscar_asignaturas para ver las disponibles."
        
        subject_id = subject_response.data[0]['id']
        subject_name_real = subject_response.data[0]['name']
        
        # Obtener usuario actual
        user_id = st.session_state.get('user_id')
        if not user_id:
            return "Error: No se pudo identificar al usuario."
        
        # Verificar si ya está inscrito en esta asignatura
        user_schedule, registered_subject_ids = get_user_schedule_data(user_id)
        
        if subject_id in registered_subject_ids:
            return f"⚠️ Ya estás inscrito en '{subject_name_real}' en otra sección."
        
        # Obtener secciones
        sections_response = supabase.table('sections').select('*').eq('subject_id', subject_id).execute()
        sections = sections_response.data
        
        if not sections:
            return f"No hay secciones disponibles para '{subject_name_real}'."
        
        result = f"📋 **Secciones disponibles para {subject_name_real}:**\n\n"
        
        for sec in sections:
            # Contar inscripciones
            registrations_count_response = supabase.table('registrations').select('id', count='exact').eq('section_id', sec['id']).execute()
            registrations_count = registrations_count_response.count or 0
            cupos_disponibles = sec['capacity'] - registrations_count
            
            # Verificar conflicto de horario
            conflicto = check_schedule_conflict_func(user_schedule, sec)
            
            result += f"**Sección {sec['section_code']}**\n"
            result += f"  - Horario: {sec['day_of_week']} de {sec['start_time']} a {sec['end_time']}\n"
            result += f"  - Cupos: {cupos_disponibles}/{sec['capacity']} disponibles\n"
            result += f"  - ID Sección: {sec['id']}\n"
            
            if conflicto:
                result += f"  - ⚠️ TOPE DE HORARIO con otra asignatura inscrita\n"
            elif cupos_disponibles == 0:
                result += f"  - ❌ SECCIÓN LLENA\n"
            else:
                result += f"  - ✅ Puedes inscribirte\n"
            
            result += "\n"
        
        return result
    except Exception as e:
        return f"Error al obtener secciones: {str(e)}"

@tool
def inscribir_seccion(section_id: int) -> str:
    """
    Inscribe al usuario en una sección específica.
    
    Args:
        section_id: ID de la sección en la que se desea inscribir
    
    Returns:
        Mensaje de confirmación o error
    """
    try:
        user_id = st.session_state.get('user_id')
        user_name = st.session_state.get('name', 'Usuario')
        
        if not user_id:
            return "Error: No se pudo identificar al usuario."
        
        # Obtener información de la sección
        section_response = supabase.table('sections').select('*, subjects(name)').eq('id', section_id).execute()
        
        if not section_response.data:
            return f"No se encontró la sección con ID {section_id}."
        
        section = section_response.data[0]
        subject_name = section['subjects']['name']
        
        # Verificar horario actual del usuario
        user_schedule, registered_subject_ids = get_user_schedule_data(user_id)
        
        # Verificar si ya está inscrito en esta asignatura
        if section['subject_id'] in registered_subject_ids:
            return f"⚠️ Ya estás inscrito en '{subject_name}' en otra sección."
        
        # Verificar conflicto de horario
        if check_schedule_conflict_func(user_schedule, section):
            return f"⚠️ No se puede inscribir: Tope de horario. Ya tienes una clase el {section['day_of_week']} entre {section['start_time']} y {section['end_time']}."
        
        # Verificar cupos
        registrations_count_response = supabase.table('registrations').select('id', count='exact').eq('section_id', section_id).execute()
        registrations_count = registrations_count_response.count or 0
        cupos_disponibles = section['capacity'] - registrations_count
        
        if cupos_disponibles <= 0:
            return f"❌ No se puede inscribir: La sección {section['section_code']} está llena."
        
        # Inscribir
        supabase.table('registrations').insert({
            'user_id': user_id, 
            'section_id': section_id
        }).execute()
        
        # Limpiar caché
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
        
        return f"✅ ¡Inscripción exitosa, {user_name}!\n\nHas sido inscrito en:\n- {subject_name}\n- Sección: {section['section_code']}\n- Horario: {section['day_of_week']} de {section['start_time']} a {section['end_time']}\n\nUsa 'ver_mi_horario' para ver tu horario actualizado."
    
    except Exception as e:
        return f"Error al inscribir: {str(e)}"

@tool
def ver_mi_horario() -> str:
    """
    Muestra el horario actual del usuario con todas sus asignaturas inscritas.
    
    Returns:
        Horario completo del usuario
    """
    try:
        user_id = st.session_state.get('user_id')
        user_name = st.session_state.get('name', 'Usuario')
        
        if not user_id:
            return "Error: No se pudo identificar al usuario."
        
        all_regs_response = supabase.table('registrations').select(
            'id, sections(section_code, day_of_week, start_time, end_time, subjects(name))'
        ).eq('user_id', user_id).execute()
        
        all_regs = all_regs_response.data
        
        if not all_regs:
            return f"📅 {user_name}, aún no tienes asignaturas inscritas."
        
        result = f"📅 **Horario de {user_name}**\n\n"
        
        # Organizar por día de la semana
        days_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado']
        schedule_by_day = {day: [] for day in days_order}
        
        for reg in all_regs:
            sec = reg.get('sections')
            if sec and sec.get('subjects'):
                day = sec['day_of_week']
                if day in schedule_by_day:
                    schedule_by_day[day].append({
                        'subject': sec['subjects']['name'],
                        'section': sec['section_code'],
                        'start': sec['start_time'],
                        'end': sec['end_time'],
                        'reg_id': reg['id']
                    })
        
        for day in days_order:
            classes = schedule_by_day[day]
            if classes:
                result += f"**{day}:**\n"
                # Ordenar por hora de inicio
                classes.sort(key=lambda x: x['start'])
                for cls in classes:
                    result += f"  • {cls['subject']} (Sección {cls['section']})\n"
                    result += f"    {cls['start']} - {cls['end']}\n"
                result += "\n"
        
        total_courses = len(all_regs)
        result += f"📊 **Total de asignaturas inscritas:** {total_courses}"
        
        return result
    
    except Exception as e:
        return f"Error al obtener horario: {str(e)}"

@tool
def anular_inscripcion(subject_name: str) -> str:
    """
    Anula la inscripción de una asignatura específica.
    
    Args:
        subject_name: Nombre de la asignatura a anular
    
    Returns:
        Mensaje de confirmación o error
    """
    try:
        user_id = st.session_state.get('user_id')
        
        if not user_id:
            return "Error: No se pudo identificar al usuario."
        
        # Buscar la inscripción
        all_regs_response = supabase.table('registrations').select(
            'id, sections(subjects(name))'
        ).eq('user_id', user_id).execute()
        
        all_regs = all_regs_response.data
        
        if not all_regs:
            return "No tienes asignaturas inscritas para anular."
        
        # Buscar la asignatura específica
        registration_to_delete = None
        for reg in all_regs:
            if reg.get('sections') and reg['sections'].get('subjects'):
                if reg['sections']['subjects']['name'].lower() == subject_name.lower():
                    registration_to_delete = reg['id']
                    break
        
        if not registration_to_delete:
            return f"No se encontró una inscripción para '{subject_name}'. Verifica el nombre exacto con 'ver_mi_horario'."
        
        # Eliminar la inscripción
        supabase.table('registrations').delete().eq('id', registration_to_delete).execute()
        
        # Limpiar caché
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
        
        return f"✅ Has anulado exitosamente la inscripción de '{subject_name}'."
    
    except Exception as e:
        return f"Error al anular inscripción: {str(e)}"

# --- CACHING DE RECURSOS DEL CHATBOT ---
@st.cache_resource
def inicializar_agente_chatbot():
    """Inicializa el agente del chatbot con herramientas de inscripción"""
    # Cargar y procesar el PDF del reglamento
    loader = PyPDFLoader("reglamento.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = loader.load_and_split(text_splitter=text_splitter)
    
    # Crear embeddings y retriever
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 7
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.7, 0.3]
    )
    
    # Crear herramienta de consulta de reglamento
    @tool
    def consultar_reglamento(pregunta: str) -> str:
        """
        Consulta el reglamento académico de Duoc UC para responder preguntas sobre normas, artículos y procedimientos.
        
        Args:
            pregunta: La pregunta sobre el reglamento académico
        
        Returns:
            Respuesta basada en el reglamento
        """
        try:
            user_name = st.session_state.get('name', 'Usuario')
            
            # Usar el retriever para obtener contexto
            relevant_docs = retriever.get_relevant_documents(pregunta)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.1)
            
            prompt_template = f"""
            Eres un asistente experto en el reglamento académico de Duoc UC.
            
            CONTEXTO DEL REGLAMENTO:
            {context}
            
            PREGUNTA DE {user_name}:
            {pregunta}
            
            INSTRUCCIONES:
            1. Responde de forma clara y concisa basándote ÚNICAMENTE en el contexto proporcionado.
            2. Si encuentras el artículo relevante, cítalo (ej. "Artículo N°30").
            3. Si la pregunta es general sobre ser "alumno nuevo", da un resumen de los puntos clave.
            4. Mantén un tono amigable y cercano.
            
            RESPUESTA:
            """
            
            response = llm.invoke(prompt_template)
            return response.content
        
        except Exception as e:
            return f"Error al consultar el reglamento: {str(e)}"
    
    # Definir las herramientas disponibles
    tools = [
        consultar_reglamento,
        buscar_asignaturas,
        ver_secciones_asignatura,
        inscribir_seccion,
        ver_mi_horario,
        anular_inscripcion
    ]
    
    # Crear el LLM
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.1)
    
    # Prompt del sistema para el agente
    system_prompt = """Eres el asistente académico de Duoc UC. Puedes ayudar a los estudiantes con:

1. **Consultas sobre el reglamento académico**: Usa la herramienta 'consultar_reglamento'
2. **Búsqueda de asignaturas**: Usa 'buscar_asignaturas'
3. **Ver secciones disponibles**: Usa 'ver_secciones_asignatura'
4. **Inscribir asignaturas**: Usa 'inscribir_seccion'
5. **Ver horario actual**: Usa 'ver_mi_horario'
6. **Anular inscripciones**: Usa 'anular_inscripcion'

INSTRUCCIONES IMPORTANTES:
- Siempre sé amigable y usa el nombre del estudiante cuando lo conozcas
- Para inscripciones, PRIMERO muestra las secciones disponibles con 'ver_secciones_asignatura'
- LUEGO pregunta al usuario cuál sección prefiere
- Solo usa 'inscribir_seccion' cuando el usuario confirme el ID de la sección
- Si hay conflictos de horario o cupos llenos, explícalo claramente
- Mantén las respuestas concisas pero completas

El usuario actual se llama: {user_name}
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # Crear el agente
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor

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
    'abcdefg123456',
    cookie_expiry_days=30
)

# --- INICIO DE LA LÓGICA DE LA APLICACIÓN ---

# --- Título Principal con Logo ---
col_title1, col_title2 = st.columns([0.1, 0.9])
with col_title1:
    st.image(LOGO_ICON_URL, width=70)
with col_title2:
    st.title("Asistente Académico Duoc UC")

# Comprobar si el usuario ya está logueado
if st.session_state["authentication_status"] is True:
    user_name = st.session_state["name"]
    user_email = st.session_state["username"]
    
    st.sidebar.image(LOGO_BANNER_URL) 
    
    if 'user_id' not in st.session_state:
        user_id_response = supabase.table('profiles').select('id').eq('email', user_email).execute()
        if user_id_response.data:
            st.session_state.user_id = user_id_response.data[0]['id']
        else:
            st.error("Error crítico: No se pudo encontrar el perfil del usuario logueado.")
            st.stop()
    
    user_id = st.session_state.user_id

    # --- Encabezado y Logout ---
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.caption(f"Conectado como: {user_name} ({user_email})")
    with col2:
        if st.button("Cerrar Sesión", use_container_width=True, key="logout_button_global"):
            authenticator.logout()
            st.session_state.clear()
            st.rerun()

    # --- NAVEGACIÓN PRINCIPAL (PESTAÑAS) ---
    tab1, tab2 = st.tabs(["💬 Chatbot Inteligente", "📋 Inscripción Manual"])

    # --- PESTAÑA 1: CHATBOT INTELIGENTE CON AGENTE ---
    with tab1:
        st.markdown(f"### ¡Hola {user_name}! 👋")
        st.markdown("""
        Puedo ayudarte con:
        - 📖 Consultas sobre el **reglamento académico**
        - 🔍 **Buscar asignaturas** disponibles
        - 📝 **Inscribirte** en asignaturas
        - 📅 Ver tu **horario** actual
        - ❌ **Anular** inscripciones
        
        Solo pregúntame lo que necesites, ¡como por ejemplo: "Quiero inscribir Matemáticas"!
        """)
        
        if st.button("🗑️ Limpiar Historial del Chat", use_container_width=True, key="clear_chat"):
            supabase.table('chat_history').delete().eq('user_id', user_id).execute()
            st.session_state.messages = []
            welcome_message = f"¡Hola {user_name}! Tu historial ha sido limpiado. ¿En qué te puedo ayudar?"
            st.session_state.messages.append({"role": "assistant", "content": welcome_message, "id": None})
            supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': welcome_message}).execute()
            st.rerun() 
        
        st.divider() 
        
        # Inicializar el agente
        agent_executor = inicializar_agente_chatbot()

        # Cargar historial de chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
            history = supabase.table('chat_history').select('id, role, message').eq('user_id', user_id).order('created_at').execute()
            for row in history.data:
                st.session_state.messages.append({"id": row['id'], "role": row['role'], "content": row['message']})
            
            if not st.session_state.messages:
                welcome_message = f"¡Hola {user_name}! Soy tu asistente académico. Puedo ayudarte con el reglamento, inscripción de asignaturas y más. ¿Qué necesitas?"
                response = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': welcome_message}).execute()
                new_message_id = response.data[0]['id'] if response.data else None
                st.session_state.messages.append({"id": new_message_id, "role": "assistant", "content": welcome_message})

        # Mostrar mensajes del historial
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Lógica de feedback
                if message["role"] == "assistant" and message["id"] is not None:
                    feedback_response = supabase.table('feedback').select('id, rating').eq('message_id', message['id']).execute()
                    existing_feedback = feedback_response.data
                    
                    if not existing_feedback:
                        col_fb1, col_fb2, col_fb_rest = st.columns([1, 1, 8])
                        with col_fb1:
                            if st.button("👍", key=f"up_{message['id']}", use_container_width=True):
                                supabase.table('feedback').insert({"message_id": message['id'], "user_id": user_id, "rating": "good"}).execute()
                                st.toast("¡Gracias por tu feedback!")
                                time.sleep(1)
                                st.rerun()
                        with col_fb2:
                            if st.button("👎", key=f"down_{message['id']}", use_container_width=True):
                                supabase.table('feedback').insert({"message_id": message['id'], "user_id": user_id, "rating": "bad"}).execute()
                                st.toast("¡Gracias! Tu feedback nos ayuda a mejorar.")
                                time.sleep(1)
                                st.rerun()
                    else:
                        if existing_feedback[0]['rating'] == 'good':
                            st.markdown("<span>Gracias por tu feedback 👍</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("<span>Gracias por tu feedback 👎</span>", unsafe_allow_html=True)

        # Procesar nueva pregunta
        if prompt := st.chat_input("Escribe tu pregunta o solicitud aquí..."):
            # Agregar mensaje del usuario
            st.session_state.messages.append({"role": "user", "content": prompt, "id": None})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            supabase.table('chat_history').insert({'user_id': user_id, 'role': 'user', 'message': prompt}).execute()
            
            # Generar respuesta del agente
            with st.chat_message("assistant"):
                with st.spinner("Pensando... 💭"):
                    try:
                        response = agent_executor.invoke({
                            "input": prompt,
                            "user_name": user_name
                        })
                        respuesta_bot = response["output"]
                        st.markdown(respuesta_bot)
                    except Exception as e:
                        respuesta_bot = f"Lo siento, hubo un error al procesar tu solicitud: {str(e)}"
                        st.error(respuesta_bot)
            
            # Guardar respuesta del bot
            response_bot_insert = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': respuesta_bot}).execute()
            new_message_id = response_bot_insert.data[0]['id'] if response_bot_insert.data else None
            
            st.session_state.messages.append({"id": new_message_id, "role": "assistant", "content": respuesta_bot})
            st.rerun()

    # --- PESTAÑA 2: INSCRIPCIÓN MANUAL (INTERFAZ VISUAL) ---
    with tab2:
        st.header("Inscripción Manual de Asignaturas")
        st.info("💡 **Tip:** También puedes inscribir asignaturas desde el chatbot. ¡Solo pídelo!")
        
        @st.cache_data(ttl=60) 
        def get_user_schedule(user_uuid):
            return get_user_schedule_data(user_uuid)

        def check_schedule_conflict(user_schedule, new_section):
            return check_schedule_conflict_func(user_schedule, new_section)

        @st.cache_data(ttl=300) 
        def get_all_subjects():
            subjects_response = supabase.table('subjects').select('id, name').order('name').execute()
            return {subj['name']: subj['id'] for subj in subjects_response.data}
        
        subjects_dict = get_all_subjects()
        
        if not subjects_dict:
             st.warning("No hay asignaturas cargadas en la base de datos.")
        else:
            selected_subject_name = st.selectbox("Selecciona una asignatura para inscribir:", options=subjects_dict.keys())
            
            if selected_subject_name:
                selected_subject_id = subjects_dict[selected_subject_name]
                sections_response = supabase.table('sections').select('*').eq('subject_id', selected_subject_id).execute()
                sections = sections_response.data
                
                if not sections:
                    st.warning("No hay secciones disponibles para esta asignatura.")
                else:
                    st.subheader(f"Secciones disponibles para {selected_subject_name}:")
                    user_schedule, registered_subject_ids = get_user_schedule(user_id)
                    
                    if selected_subject_id in registered_subject_ids:
                        st.error("Ya tienes esta asignatura inscrita en otra sección.")
                    else:
                        for sec in sections:
                            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                            registrations_count_response = supabase.table('registrations').select('id', count='exact').eq('section_id', sec['id']).execute()
                            registrations_count = registrations_count_response.count
                            cupos_disponibles = sec['capacity'] - (registrations_count if registrations_count else 0)
                            
                            col1.text(f"Sección {sec['section_code']}")
                            col2.text(f"{sec['day_of_week']} de {sec['start_time']} a {sec['end_time']}")
                            col3.text(f"{cupos_disponibles} de {sec['capacity']} cupos disponibles")
                            
                            with col4:
                                if cupos_disponibles > 0:
                                    if st.button("Inscribir", key=sec['id']):
                                        if check_schedule_conflict(user_schedule, sec):
                                            st.error(f"¡Tope de horario! Ya tienes una clase el {sec['day_of_week']} a esa hora.")
                                        else:
                                            try:
                                                supabase.table('registrations').insert({'user_id': user_id, 'section_id': sec['id']}).execute()
                                                st.success(f"¡Inscrito en la sección {sec['section_code']}!")
                                                st.cache_data.clear() 
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Error al inscribir: {e}")
                                else:
                                    st.button("Llena", disabled=True, key=sec['id'])

        st.divider()
        st.subheader(f"Horario Actual de {user_name}")
        current_schedule_info, _ = get_user_schedule(user_id) 
        if not current_schedule_info:
            st.info("Aún no tienes asignaturas inscritas.")
        else:
            all_regs_response = supabase.table('registrations').select('id, sections(subject_id, section_code, day_of_week, start_time, end_time, subjects(name))').eq('user_id', user_id).execute()
            all_regs = all_regs_response.data
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            col1.write("**Asignatura**")
            col2.write("**Día**")
            col3.write("**Horario**")
            col4.write("**Acción**")
            for reg in all_regs:
                sec = reg['sections']
                if sec and sec['subjects']:
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    col1.write(f"{sec['subjects']['name']} (Sección {sec['section_code']})")
                    col2.write(sec['day_of_week'])
                    col3.write(f"{sec['start_time']} - {sec['end_time']}")
                    with col4:
                        if st.button("Anular", key=f"del_{reg['id']}", type="primary", use_container_width=True):
                            supabase.table('registrations').delete().eq('id', reg['id']).execute()
                            st.success(f"Has anulado {sec['subjects']['name']}.")
                            st.cache_data.clear() 
                            st.rerun() 

# Si el usuario NO está logueado, mostrar Login y Registro
else:
    authenticator.login(location='main')
    
    if st.session_state["authentication_status"] is False:
        st.error('Email o contraseña incorrecta')
    elif st.session_state["authentication_status"] is None:
        st.info('Por favor, ingresa tu email y contraseña. ¿Nuevo usuario? Regístrate en la barra lateral.')

    st.markdown("---")
    
    # Widget de "Olvidé Contraseña"
    st.subheader("¿Olvidaste tu contraseña?")
    with st.form(key="forgot_password_form", clear_on_submit=True):
        email_olvidado = st.text_input("Ingresa tu email de registro")
        submit_button = st.form_submit_button(label="Enviar enlace de recuperación")

        if submit_button:
            if not email_olvidado:
                st.error("Por favor, ingresa un email.")
            else:
                try:
                    redirect_url = "https://chatbot-duoc1.streamlit.app"
                    supabase.auth.reset_password_for_email(email_olvidado, options={'redirect_to': redirect_url})
                    st.success("¡Email de recuperación enviado! Revisa tu bandeja de entrada.")
                    time.sleep(3)
                except Exception as e:
                    st.error(f"Error al enviar el email: {e}")

    # Formulario de Registro
    with st.sidebar:
        st.image(LOGO_BANNER_URL)
        st.subheader("¿Nuevo Usuario? Regístrate")
        with st.form(key="register_form", clear_on_submit=True):
            name_reg = st.text_input("Nombre Completo")
            email_reg = st.text_input("Email")
            password_reg = st.text_input("Contraseña", type="password")
            confirm_password_reg = st.text_input("Confirmar Contraseña", type="password")
            submit_button = st.form_submit_button(label="Registrarse")

            if submit_button:
                if not name_reg: 
                    st.error("Por favor, ingresa tu nombre.")
                elif not email_reg: 
                    st.error("Por favor, ingresa tu email.")
                elif password_reg != confirm_password_reg: 
                    st.error("Las contraseñas no coinciden.")
                elif len(password_reg) < 6: 
                    st.error("La contraseña debe tener al menos 6 caracteres.")
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