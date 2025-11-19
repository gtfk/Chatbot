# Versión 8.0 - Diseño con Pestañas para Login/Registro
import streamlit as st
# Importaciones compatibles con langchain==0.1.20
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
    st.error("Faltan claves en los Secrets de Streamlit (GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY).")
    st.stop()

# --- INICIALIZAR SUPABASE ---
@st.cache_resource
def init_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase_client()

# --- CACHING DEL CHATBOT ---
@st.cache_resource
def inicializar_cadena():
    if not os.path.exists("reglamento.pdf"):
        st.error("No se encontró el archivo 'reglamento.pdf' en el repositorio.")
        st.stop()
        
    loader = PyPDFLoader("reglamento.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = loader.load_and_split(text_splitter=text_splitter)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 7
    
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5]
    )

    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.1)

    prompt_template = """
    INSTRUCCIÓN PRINCIPAL: Responde SIEMPRE en español, con un tono amigable y cercano.
    PERSONAJE: Eres un asistente experto en el reglamento académico de Duoc UC. Estás hablando con un estudiante llamado {user_name}.
    
    SI LA PREGUNTA ES GENERAL (ej. "soy nuevo", "qué debo saber"):
    Responde SOLO con este resumen:
    "¡Hola {user_name}! Como alumno nuevo, esto es lo clave:
    1. Asistencia (Art. 30): 70% mínimo en teóricas y prácticas para aprobar.
    2. Notas (Art. 37): Nota mínima 4.0.
    3. Reprobación (Art. 39): Por nota bajo 4.0 o asistencia bajo 70%."

    SI LA PREGUNTA ES ESPECÍFICA:
    Usa el contexto para responder. Si no sabes, di que no sabes.
    
    CONTEXTO: {context}
    PREGUNTA DE {user_name}: {input}
    RESPUESTA:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# --- FUNCIONES AUXILIARES DE INSCRIPCIÓN ---
@st.cache_data(ttl=60) 
def get_user_schedule(user_uuid):
    user_regs = supabase.table('registrations').select('section_id').eq('user_id', user_uuid).execute().data
    if not user_regs: return [], []
    
    section_ids = [reg['section_id'] for reg in user_regs]
    if not section_ids: return [], []

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

# --- LÓGICA DE AUTENTICACIÓN ---
def fetch_all_users():
    try:
        response = supabase.table('profiles').select("email, full_name, password_hash").execute()
        users = response.data
        if not users: return {'usernames': {}}
        credentials = {'usernames': {}}
        for user in users:
            credentials['usernames'][user['email']] = {
                'email': user['email'],
                'name': user['full_name'],
                'password': user['password_hash']
            }
        return credentials
    except Exception:
        return {'usernames': {}}

credentials = fetch_all_users()
authenticator = stauth.Authenticate(
    credentials,
    'chatbot_duoc_cookie',
    'clave_secreta_random_123',
    cookie_expiry_days=30
)

# --- INTERFAZ PRINCIPAL ---

# Título con Logo
col_title1, col_title2 = st.columns([0.15, 0.85])
with col_title1:
    st.image(LOGO_ICON_URL, width=100)
with col_title2:
    st.title("Asistente Académico Duoc UC")

# --- LÓGICA SI ESTÁ LOGUEADO ---
if st.session_state["authentication_status"]:
    user_name = st.session_state["name"]
    user_email = st.session_state["username"]
    
    # Sidebar solo con logo y logout
    st.sidebar.image(LOGO_BANNER_URL)
    if st.sidebar.button("Cerrar Sesión", use_container_width=True):
        authenticator.logout()
        st.session_state.clear()
        st.rerun()
    
    # Cargar ID de usuario
    if 'user_id' not in st.session_state:
        data = supabase.table('profiles').select('id').eq('email', user_email).execute()
        if data.data:
            st.session_state.user_id = data.data[0]['id']
    
    user_id = st.session_state.user_id

    # Pestañas de la App
    tab1, tab2 = st.tabs(["Chatbot", "Inscripción"])

    # --- PESTAÑA 1: CHATBOT ---
    with tab1:
        st.caption(f"👋 Hola, **{user_name}**")
        if st.button("Limpiar Historial", use_container_width=True):
            supabase.table('chat_history').delete().eq('user_id', user_id).execute()
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        chain = inicializar_cadena()

        # Cargar historial
        if "messages" not in st.session_state:
            st.session_state.messages = []
            history = supabase.table('chat_history').select('id, role, message').eq('user_id', user_id).order('created_at').execute()
            for row in history.data:
                st.session_state.messages.append({"id": row['id'], "role": row['role'], "content": row['message']})
            
            if not st.session_state.messages:
                welcome = f"¡Hola {user_name}! Soy tu asistente. ¿En qué te ayudo?"
                res = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': welcome}).execute()
                if res.data:
                    st.session_state.messages.append({"id": res.data[0]['id'], "role": "assistant", "content": welcome})

        # Mostrar mensajes
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Feedback
                if message["role"] == "assistant" and message.get("id"):
                    fb = supabase.table('feedback').select('rating').eq('message_id', message['id']).execute().data
                    if not fb:
                        c1, c2, _ = st.columns([1, 1, 8])
                        if c1.button("👍", key=f"up_{message['id']}"):
                            supabase.table('feedback').insert({"message_id": message['id'], "user_id": user_id, "rating": "good"}).execute()
                            st.rerun()
                        if c2.button("👎", key=f"down_{message['id']}"):
                            supabase.table('feedback').insert({"message_id": message['id'], "user_id": user_id, "rating": "bad"}).execute()
                            st.rerun()
                    else:
                        st.caption(f"Feedback: {'👍' if fb[0]['rating'] == 'good' else '👎'}")

        # Input
        if prompt := st.chat_input("Pregunta..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            supabase.table('chat_history').insert({'user_id': user_id, 'role': 'user', 'message': prompt}).execute()
            
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    response = chain.invoke({"input": prompt, "user_name": user_name})
                    st.markdown(response["answer"])
            
            res_bot = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': response["answer"]}).execute()
            if res_bot.data:
                st.session_state.messages.append({"id": res_bot.data[0]['id'], "role": "assistant", "content": response["answer"]})
            st.rerun()

    # --- PESTAÑA 2: INSCRIPCIÓN ---
    with tab2:
        st.header("Inscripción de Asignaturas")
        subjects_dict = get_all_subjects()
        
        if not subjects_dict:
            st.info("No hay asignaturas cargadas.")
        else:
            sel_subj = st.selectbox("Asignatura:", options=subjects_dict.keys())
            if sel_subj:
                s_id = subjects_dict[sel_subj]
                sections = supabase.table('sections').select('*').eq('subject_id', s_id).execute().data
                
                if not sections:
                    st.info("No hay secciones.")
                else:
                    user_sch, reg_ids = get_user_schedule(user_id)
                    if s_id in reg_ids:
                        st.success("Ya inscrita.")
                    else:
                        for sec in sections:
                            c1, c2, c3, c4 = st.columns([2,2,2,2])
                            regs = supabase.table('registrations').select('id', count='exact').eq('section_id', sec['id']).execute().count
                            cupos = sec['capacity'] - (regs if regs else 0)
                            
                            c1.write(f"**{sec['section_code']}**")
                            c2.write(f"{sec['day_of_week']}")
                            c3.write(f"{sec['start_time']} - {sec['end_time']}")
                            
                            if c4.button(f"Inscribir ({cupos})", key=sec['id'], disabled=cupos<=0):
                                if check_schedule_conflict(user_schedule, sec):
                                    st.error("Tope de horario")
                                else:
                                    supabase.table('registrations').insert({'user_id': user_id, 'section_id': sec['id']}).execute()
                                    st.toast("Inscrito")
                                    time.sleep(1)
                                    st.rerun()
        
        st.divider()
        st.subheader("Tu Horario")
        regs = supabase.table('registrations').select('id, sections(section_code, day_of_week, start_time, end_time, subjects(name))').eq('user_id', user_id).execute().data
        if regs:
            for r in regs:
                s = r['sections']
                if s:
                    c1, c2, c3, c4 = st.columns([3,2,2,1])
                    c1.write(f"{s['subjects']['name']} ({s['section_code']})")
                    c2.write(s['day_of_week'])
                    c3.write(f"{s['start_time']} - {s['end_time']}")
                    if c4.button("Anular", key=f"del_{r['id']}"):
                        supabase.table('registrations').delete().eq('id', r['id']).execute()
                        st.rerun()
        else:
            st.info("Sin inscripciones.")

# --- LÓGICA SI NO ESTÁ LOGUEADO ---
elif st.session_state["authentication_status"] is False:
    st.error('Usuario o contraseña incorrectos')
elif st.session_state["authentication_status"] is None:
    st.info('Bienvenido. Por favor inicia sesión o regístrate.')

if not st.session_state["authentication_status"]:
    
    # Pestañas PRINCIPALES para Login y Registro
    tab_login, tab_reg = st.tabs(["Iniciar Sesión", "Registrarse"])
    
    with tab_login:
        authenticator.login(location='main')
        st.markdown("---")
        with st.expander("¿Olvidaste tu contraseña?"):
            rec_email = st.text_input("Ingresa tu email para recuperar")
            if st.button("Enviar correo de recuperación"):
                if rec_email:
                    try:
                        supabase.auth.reset_password_for_email(rec_email, options={
                            'redirect_to': 'https://chatbot-duoc.streamlit.app'
                        })
                        st.success("Correo enviado. Revisa tu bandeja de entrada.")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Ingresa un email.")

    with tab_reg:
         with st.form("registro_form"):
            st.subheader("Crear Cuenta Nueva")
            new_name = st.text_input("Nombre Completo")
            new_email = st.text_input("Email Institucional")
            new_pass = st.text_input("Contraseña", type="password", help="Mínimo 6 caracteres")
            new_pass_confirm = st.text_input("Confirmar Contraseña", type="password")
            
            if st.form_submit_button("Registrarse"):
                if new_name and new_email and len(new_pass) >= 6:
                    if new_pass != new_pass_confirm:
                        st.error("Las contraseñas no coinciden.")
                    else:
                        try:
                            hasher = stauth.Hasher()
                            hashed = hasher.hash(new_pass)
                            # Guardamos en profiles
                            supabase.table('profiles').insert({
                                'full_name': new_name, 'email': new_email, 'password_hash': hashed
                            }).execute()
                            st.success("¡Cuenta creada con éxito! Ve a la pestaña 'Iniciar Sesión'.")
                        except Exception as e:
                            if 'duplicate' in str(e):
                                st.error("Ese email ya está registrado.")
                            else:
                                st.error(f"Error: {e}")
                else:
                    st.error("Por favor completa todos los campos correctamente.")