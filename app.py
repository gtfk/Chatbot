# Versión 7.1 (Estable) - Corregido el NameError (variable mal escrita)
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
from datetime import time as dt_time

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
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

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

col_title1, col_title2 = st.columns([0.1, 0.9])
with col_title1:
    st.image(LOGO_ICON_URL, width=70)
with col_title2:
    st.title("Asistente Académico Duoc UC")

# 3. Comprobar si el usuario ya está logueado
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

    # --- NAVEGACIÓN PRINCIPAL ---
    tab1, tab2 = st.tabs(["Chatbot de Reglamento", "Inscripción de Asignaturas"])

    # --- PESTAÑA 1: CHATBOT DE REGLAMENTO ---
    with tab1:
        if st.button("Limpiar Historial del Chat", use_container_width=True, key="clear_chat"):
            supabase.table('chat_history').delete().eq('user_id', user_id).execute()
            st.session_state.messages = []
            welcome_message = f"¡Hola {user_name}! Tu historial ha sido limpiado. ¿En qué te puedo ayudar?"
            st.session_state.messages.append({"role": "assistant", "content": welcome_message, "id": None})
            supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': welcome_message}).execute()
            st.rerun() 
        
        st.divider() 
        
        retrieval_chain = inicializar_cadena()

        # Cargar historial de chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
            history = supabase.table('chat_history').select('id, role, message').eq('user_id', user_id).order('created_at').execute()
            for row in history.data:
                st.session_state.messages.append({"id": row['id'], "role": row['role'], "content": row['message']})
            
            if not st.session_state.messages:
                welcome_message = f"¡Hola {user_name}! Soy tu asistente del reglamento académico. ¿En qué te puedo ayudar hoy?"
                response = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': welcome_message}).execute()
                new_message_id = response.data[0]['id'] if response.data else None
                st.session_state.messages.append({"id": new_message_id, "role": "assistant", "content": welcome_message})

        # Mostrar mensajes del historial
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # --- LÓGICA DE FEEDBACK ---
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
        if prompt := st.chat_input("¿Qué duda tienes sobre el reglamento?"):
            user_message_data = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message_data)
            with st.chat_message("user"):
                st.markdown(prompt)
            
            response_user_insert = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'user', 'message': prompt}).execute()
            
            with st.chat_message("assistant"):
                with st.spinner("Pensando... 💭"):
                    response = retrieval_chain.invoke({"input": prompt, "user_name": user_name })
                    respuesta_bot = response["answer"]
                    st.markdown(respuesta_bot)
            
            response_bot_insert = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': respuesta_bot}).execute()
            
            # --- CORRECCIÓN AQUÍ (Línea 236 y 238) ---
            # Usamos el mismo nombre de variable
            new_bot_message_id = response_bot_insert.data[0]['id'] if response_bot_insert.data else None
            st.session_state.messages.append({"id": new_bot_message_id, "role": "assistant", "content": respuesta_bot})
            # --- FIN DE LA CORRECCIÓN ---
            
            st.rerun()

    # --- PESTAÑA 2: INSCRIPCIÓN DE ASIGNATURAS ---
    with tab2:
        # ... (La lógica de inscripción es idéntica, la dejo igual por brevedad pero debe ir aquí) ...
        st.header("Inscripción de Asignaturas")
        @st.cache_data(ttl=60) 
        def get_user_schedule(user_uuid):
            user_regs = supabase.table('registrations').select('section_id').eq('user_id', user_uuid).execute().data
            if not user_regs: return [], []
            section_ids = [reg['section_id'] for reg in user_regs]
            schedule_data = supabase.table('sections').select('subject_id, day_of_week, start_time, end_time').in_('id', section_ids).execute().data
            schedule = []
            registered_subject_ids = []
            for sec in schedule_data:
                schedule.append({"day": sec['day_of_week'], "start": dt_time.fromisoformat(sec['start_time']), "end": dt_time.fromisoformat(sec['end_time'])})
                registered_subject_ids.append(sec['subject_id'])
            return schedule, registered_subject_ids

        def check_schedule_conflict(user_schedule, new_section):
            new_day = new_section['day_of_week']
            new_start = dt_time.fromisoformat(new_section['start_time'])
            new_end = dt_time.fromisoformat(new_section['end_time'])
            for scheduled in user_schedule:
                if scheduled['day'] == new_day:
                    if max(scheduled['start'], new_start) < min(scheduled['end'], new_end):
                        return True 
            return False 

        @st.cache_data(ttl=300) 
        def get_all_subjects():
            subjects_response = supabase.table('subjects').select('id, name').order('name').execute()
            return {subj['name']: subj['id'] for subj in subjects_response.data}
        
        subjects_dict = get_all_subjects()
        
        if not subjects_dict:
             st.warning("No hay asignaturas cargadas en la base de datos. Por favor, ejecuta el script de 'seeding' de Colab.")
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

# 4. Si el usuario NO está logueado, mostrar Login y Registro
else:
    authenticator.login(location='main')
    
    if st.session_state["authentication_status"] is False:
        st.error('Email o contraseña incorrecta')
    elif st.session_state["authentication_status"] is None:
        st.info('Por favor, ingresa tu email y contraseña. ¿Nuevo usuario? Registrate en la barra lateral.')

    st.markdown("---")
    
    # Formulario de "Olvidé mi contraseña"
    try:
        if authenticator.forgot_password(form_name="¿Olvidaste tu contraseña?", location='main'):
            email_olvidado = st.session_state.email
            try:
                redirect_url = "https://chatbot-duoc1.streamlit.app" 
                supabase.auth.reset_password_for_email(email_olvidado, options={'redirect_to': redirect_url})
                st.success("¡Email de recuperación enviado! Revisa tu bandeja de entrada.")
                time.sleep(3)
            except Exception as e:
                st.error(f"Error al enviar el email: {e}")
    except Exception as e:
        st.error(f"Error en el proceso de contraseña olvidada: {e}")

    # Registro
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
                if not name_reg: st.error("Por favor, ingresa tu nombre.")
                elif not email_reg: st.error("Por favor, ingresa tu email.")
                elif password_reg != confirm_password_reg: st.error("Las contraseñas no coinciden.")
                elif len(password_reg) < 6: st.error("La contraseña debe tener al menos 6 caracteres.")
                else:
                    try:
                        hasher = stauth.Hasher()
                        hashed_password = hasher.hash(password_reg)
                        insert_response = supabase.table('profiles').insert({'full_name': name_reg, 'email': email_reg, 'password_hash': hashed_password}).execute()
                        if insert_response.data:
                            st.success('¡Usuario registrado! Ahora puedes iniciar sesión.')
                            time.sleep(2) 
                        else:
                            st.error('Error al registrar el usuario.')
                    except Exception as e:
                        if 'duplicate key value' in str(e): st.error("Error: Ese email ya está registrado.")
                        else: st.error(f"Error en el registro: {e}")