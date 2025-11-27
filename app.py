# Versión 25.0 (FINAL: Traducción Completa Chips + Excel + Filtros + Base Estable)
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
import bcrypt
import pandas as pd

# --- URLs DE LOGOS ---
LOGO_BANNER_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Logo_DuocUC.svg/2560px-Logo_DuocUC.svg.png"
LOGO_ICON_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlve2kMlU53cq9Tl0DMxP0Ffo0JNap2dXq4q_uSdf4PyFZ9uraw7MU5irI6mA-HG8byNI&usqp=CAU"

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Chatbot Duoc UC",
    page_icon=LOGO_ICON_URL,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CARGAR CSS DESDE ARCHIVO EXTERNO ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"⚠️ No se encontró el archivo {file_name}. Asegúrate de que esté en la misma carpeta que app.py.")

# Cargar estilos visuales
load_css("styles.css")

# --- DICCIONARIO DE TRADUCCIONES (ACTUALIZADO CON CHIPS) ---
TEXTS = {
    "es": {
        "label": "Español 🇨🇱",
        "title": "Asistente Académico Duoc UC",
        "sidebar_lang": "Idioma / Language",
        "login_success": "Usuario:",
        "logout_btn": "Cerrar Sesión",
        "tab1": "💬 Chatbot Reglamento",
        "tab2": "📅 Inscripción de Asignaturas",
        "tab3": "🔐 Admin / Auditoría",
        "login_title": "Iniciar Sesión",
        "login_user": "Correo Institucional",
        "login_pass": "Contraseña",
        "login_btn": "Ingresar",
        "login_failed": "❌ Credenciales inválidas",
        "login_welcome": "¡Bienvenido al Asistente!",
        "chat_clear_btn": "🧹 Limpiar Conversación",
        "chat_cleaning": "Procesando solicitud...",
        "chat_cleaned": "¡Historial limpiado!",
        "chat_welcome": "¡Hola **{name}**! 👋 Soy tu asistente virtual de Duoc UC. Pregúntame sobre el reglamento, asistencia o notas.",
        "chat_welcome_clean": "¡Hola **{name}**! El historial ha sido archivado. ¿En qué más te ayudo?",
        "chat_placeholder": "Ej: ¿Con qué nota apruebo el ramo?",
        "chat_thinking": "Consultando reglamento...",
        "feedback_thanks": "¡Gracias por tu feedback! 👍",
        "feedback_report_sent": "Reporte enviado.",
        "feedback_modal_title": "¿Qué podemos mejorar?",
        "feedback_modal_placeholder": "Ej: La información sobre asistencia no es precisa...",
        "btn_send": "Enviar Comentario",
        "btn_cancel": "Omitir",
        "enroll_title": "Toma de Ramos 2025",
        "filter_career": "📂 Filtrar por Carrera:",
        "filter_sem": "⏳ Filtrar por Semestre:",
        "filter_all": "Todas las Carreras",
        "filter_all_m": "Todos los Semestres",
        "reset_btn": "🔄 Limpiar Filtros",
        "search_label": "📚 Buscar Asignatura:",
        "search_placeholder": "Escribe el nombre del ramo...",
        "sec_title": "Secciones Disponibles para:",
        "btn_enroll": "Inscribir",
        "btn_full": "Sin Cupos",
        "msg_enrolled": "✅ ¡Asignatura inscrita exitosamente!",
        "msg_conflict": "⛔ Error: Tope de Horario detectado",
        "msg_already": "ℹ️ Ya estás inscrito en esta asignatura.",
        "my_schedule": "Tu Carga Académica",
        "no_schedule": "No tienes ramos inscritos.",
        "btn_drop": "Anular Ramo",
        "msg_dropped": "Asignatura eliminada de tu carga.",
        "admin_title": "Panel de Control (Admin)",
        "admin_pass_label": "Clave de Acceso:",
        "admin_success": "Acceso Autorizado",
        "admin_info": "Registro de interacciones y feedback negativo.",
        "admin_update_btn": "🔄 Refrescar Datos",
        "col_date": "Fecha/Hora",
        "col_status": "Estado",
        "col_q": "Pregunta Estudiante",
        "col_a": "Respuesta IA",
        "col_val": "Eval",
        "col_com": "Detalle",
        "reg_header": "Crear Cuenta Alumno",
        "reg_name": "Nombre y Apellido",
        "reg_email": "Correo Duoc",
        "reg_pass": "Crear Contraseña",
        "reg_btn": "Registrarse",
        "reg_success": "¡Cuenta creada! Accede desde el Login.",
        "auth_error": "Verifica tus datos.",
        # --- NUEVAS TRADUCCIONES PARA CHIPS ---
        "sug_header": "💡 **¿No sabes qué preguntar? Prueba con esto:**",
        "sug_btn1": "📋 Justificar Inasistencia",
        "sug_query1": "¿Cómo justifico una inasistencia?",
        "sug_btn2": "🎓 Requisitos Titulación",
        "sug_query2": "¿Cuáles son los requisitos para titularme?",
        "sug_btn3": "📅 Fechas Exámenes",
        "sug_query3": "¿Cuándo son los exámenes transversales?",
        "system_prompt": """
        INSTRUCCIÓN: Responde en Español formal pero cercano.
        ROL: Eres un coordinador académico de Duoc UC.
        """
    },
    "en": {
        "label": "English 🇺🇸",
        "title": "Duoc UC Academic Assistant",
        "sidebar_lang": "Language / Idioma",
        "login_success": "User:",
        "logout_btn": "Log Out",
        "tab1": "💬 Rulebook Chat",
        "tab2": "📅 Course Enrollment",
        "tab3": "🔐 Admin / Audit",
        "login_title": "Student Login",
        "login_user": "Institutional Email",
        "login_pass": "Password",
        "login_btn": "Login",
        "login_failed": "❌ Invalid credentials",
        "login_welcome": "Welcome to the Assistant!",
        "chat_clear_btn": "🧹 Clear Conversation",
        "chat_cleaning": "Processing...",
        "chat_cleaned": "History cleared!",
        "chat_welcome": "Hello **{name}**! 👋 I'm your Duoc UC virtual assistant. Ask me about regulations, attendance, or grades.",
        "chat_welcome_clean": "Hello **{name}**! History archived. Can I help with anything else?",
        "chat_placeholder": "Ex: What is the passing grade?",
        "chat_thinking": "Consulting rulebook...",
        "feedback_thanks": "Thanks for your feedback! 👍",
        "feedback_report_sent": "Report sent.",
        "feedback_modal_title": "What went wrong?",
        "feedback_modal_placeholder": "Ex: The information is inaccurate...",
        "btn_send": "Send Comment",
        "btn_cancel": "Skip",
        "enroll_title": "Course Registration 2025",
        "filter_career": "📂 Filter by Career:",
        "filter_sem": "⏳ Filter by Semester:",
        "filter_all": "All Careers",
        "filter_all_m": "All Semesters",
        "reset_btn": "🔄 Clear Filters",
        "search_label": "📚 Search Subject:",
        "search_placeholder": "Type subject name...",
        "sec_title": "Available Sections for:",
        "btn_enroll": "Enroll",
        "btn_full": "Full",
        "msg_enrolled": "✅ Subject enrolled successfully!",
        "msg_conflict": "⛔ Error: Schedule Conflict",
        "msg_already": "ℹ️ You are already enrolled.",
        "my_schedule": "Your Academic Load",
        "no_schedule": "No subjects enrolled.",
        "btn_drop": "Drop Course",
        "msg_dropped": "Subject removed from load.",
        "admin_title": "Control Panel (Admin)",
        "admin_pass_label": "Access Key:",
        "admin_success": "Access Granted",
        "admin_info": "Log of interactions and negative feedback.",
        "admin_update_btn": "🔄 Refresh Data",
        "col_date": "Date/Time",
        "col_status": "Status",
        "col_q": "Student Question",
        "col_a": "AI Answer",
        "col_val": "Rate",
        "col_com": "Detail",
        "reg_header": "Create Student Account",
        "reg_name": "Full Name",
        "reg_email": "Duoc Email",
        "reg_pass": "Create Password",
        "reg_btn": "Register",
        "reg_success": "Account created! Please login.",
        "auth_error": "Check your credentials.",
        # --- NUEVAS TRADUCCIONES PARA CHIPS ---
        "sug_header": "💡 **Don't know what to ask? Try this:**",
        "sug_btn1": "📋 Justify Absence",
        "sug_query1": "How do I justify an absence?",
        "sug_btn2": "🎓 Graduation Reqs",
        "sug_query2": "What are the requirements for graduation?",
        "sug_btn3": "📅 Exam Dates",
        "sug_query3": "When are the transversal exams?",
        "system_prompt": """
        INSTRUCTION: Respond in English, formal but friendly.
        ROLE: You are an academic coordinator at Duoc UC.
        """
    }
}

# --- CARGA DE CLAVES ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "DUOC2025")

if not GROQ_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Error: Faltan claves de API.")
    st.stop()

# --- SUPABASE ---
@st.cache_resource
def init_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase_client()

# --- STREAMING ---
def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# --- CHATBOT ENGINE ---
@st.cache_resource
def inicializar_cadena(language_code):
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
    
    base_instruction = TEXTS[language_code]["system_prompt"]
    
    prompt_template = base_instruction + """
    RULES:
    1. Address {user_name} by name.
    2. Be clear and concise.
    3. Base answer ONLY on context.
    4. Cite the article (e.g. "Article N°30").

    CONTEXT:
    {context}
    QUESTION FROM {user_name}:
    {input}
    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# --- FETCH USERS ---
def fetch_all_users():
    try:
        response = supabase.table('profiles').select("email, full_name, password_hash").execute()
        if not response.data: return {}
        users_dict = {u['email']: u for u in response.data}
        return users_dict
    except: return {}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"""
        <div class="sidebar-logo-container">
            <img src="{LOGO_BANNER_URL}" style="width: 100%; max-width: 180px;">
        </div>
    """, unsafe_allow_html=True)
    
    lang_option = st.selectbox("🌐 Language / Idioma", ["Español 🇨🇱", "English 🇺🇸"], format_func=lambda x: TEXTS["es" if "Español" in x else "en"]["label"])
    if "Español" in lang_option: lang_code = "es"
    else: lang_code = "en"
    t = TEXTS[lang_code]

# --- CABECERA ---
col_title1, col_title2 = st.columns([0.1, 0.9])
with col_title1: st.image(LOGO_ICON_URL, width=70)
with col_title2: st.title(t["title"])

# --- ESTADO DE AUTENTICACIÓN ---
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

# ==========================================
# APP PRINCIPAL
# ==========================================
if st.session_state["authentication_status"] is True:
    user_name = st.session_state["name"]
    user_email = st.session_state["username"]
    
    if 'user_id' not in st.session_state:
        user_id_response = supabase.table('profiles').select('id').eq('email', user_email).execute()
        if user_id_response.data: st.session_state.user_id = user_id_response.data[0]['id']
        else: st.stop()
    user_id = st.session_state.user_id

    c1, c2 = st.columns([0.8, 0.2])
    c1.caption(f"{t['login_success']} {user_name} ({user_email})")
    if c2.button(t["logout_btn"], use_container_width=True):
        st.session_state["authentication_status"] = None
        st.session_state.clear()
        st.rerun()

    tab1, tab2, tab3 = st.tabs([t["tab1"], t["tab2"], t["tab3"]])

    # --- TAB 1: CHATBOT ---
    with tab1:
        if st.button(t["chat_clear_btn"], use_container_width=True, key="clear_chat"):
            with st.spinner(t["chat_cleaning"]):
                try:
                    supabase.table('chat_history').update({'is_visible': False}).eq('user_id', user_id).execute()
                    st.session_state.messages = []
                    welcome_msg = t["chat_welcome_clean"].format(name=user_name)
                    res = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': welcome_msg}).execute()
                    if res.data:
                        st.session_state.messages.append({"id": res.data[0]['id'], "role": "assistant", "content": welcome_msg})
                    keys_to_remove = [k for k in st.session_state.keys() if k.startswith("show_reason_")]
                    for k in keys_to_remove: del st.session_state[k]
                    st.success(t["chat_cleaned"])
                    time.sleep(1)
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")
        
        st.divider()
        retrieval_chain = inicializar_cadena(lang_code)

        if "messages" not in st.session_state:
            st.session_state.messages = []
            history = supabase.table('chat_history').select('id, role, message').eq('user_id', user_id).eq('is_visible', True).order('created_at').execute()
            for row in history.data:
                st.session_state.messages.append({"id": row['id'], "role": row['role'], "content": row['message']})
            if not st.session_state.messages:
                welcome_msg = t["chat_welcome"].format(name=user_name)
                res = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': welcome_msg}).execute()
                if res.data:
                    st.session_state.messages.append({"id": res.data[0]['id'], "role": "assistant", "content": welcome_msg})

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg["id"]:
                    col_fb1, col_fb2, _ = st.columns([1,1,8])
                    if col_fb1.button("👍", key=f"up_{msg['id']}"):
                        supabase.table('feedback').insert({"message_id": msg['id'], "user_id": user_id, "rating": "good"}).execute()
                        st.toast(t["feedback_thanks"])
                    reason_key = f"show_reason_{msg['id']}"
                    if col_fb2.button("👎", key=f"down_{msg['id']}"): st.session_state[reason_key] = True
                    if st.session_state.get(reason_key, False):
                        with st.form(key=f"form_{msg['id']}", enter_to_submit=False):
                            st.write(t["feedback_modal_title"])
                            comment_text = st.text_area("...", placeholder=t["feedback_modal_placeholder"], label_visibility="collapsed")
                            c_sub1, c_sub2 = st.columns(2)
                            if c_sub1.form_submit_button(t["btn_send"]):
                                supabase.table('feedback').insert({"message_id": msg['id'], "user_id": user_id, "rating": "bad", "comment": comment_text}).execute()
                                st.toast(t["feedback_report_sent"])
                                st.session_state[reason_key] = False
                                st.rerun()
                            if c_sub2.form_submit_button(t["btn_cancel"]):
                                st.session_state[reason_key] = False
                                st.rerun()

        # --- CHIPS TRADUCIBLES Y FUNCIONALES ---
        if not st.session_state.messages or (len(st.session_state.messages) == 1 and st.session_state.messages[0]['role'] == 'assistant'):
            st.markdown(t["sug_header"])
            col_sug1, col_sug2, col_sug3 = st.columns(3)
            sugerencia = None
            # Botones ahora usan el diccionario 't' para el texto
            if col_sug1.button(t["sug_btn1"]): sugerencia = t["sug_query1"]
            if col_sug2.button(t["sug_btn2"]): sugerencia = t["sug_query2"]
            if col_sug3.button(t["sug_btn3"]): sugerencia = t["sug_query3"]
            
            if sugerencia:
                # 1. Guardar mensaje usuario
                st.session_state.messages.append({"role": "user", "content": sugerencia})
                supabase.table('chat_history').insert({'user_id': user_id, 'role': 'user', 'message': sugerencia}).execute()

                # 2. Generar respuesta IA INMEDIATAMENTE
                with st.spinner(t["chat_thinking"]):
                    try:
                        response = retrieval_chain.invoke({"input": sugerencia, "user_name": user_name})
                        resp = response["answer"]
                        
                        # 3. Guardar respuesta IA
                        res_bot = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': resp}).execute()
                        st.session_state.messages.append({"id": res_bot.data[0]['id'], "role": "assistant", "content": resp})
                    except Exception as e:
                        st.error(f"Error generando respuesta: {e}")
                # 4. Recargar para mostrar
                st.rerun()

        if prompt := st.chat_input(t["chat_placeholder"]):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            supabase.table('chat_history').insert({'user_id': user_id, 'role': 'user', 'message': prompt}).execute()
            with st.chat_message("assistant"):
                with st.spinner(t["chat_thinking"]):
                    response = retrieval_chain.invoke({"input": prompt, "user_name": user_name})
                    resp = response["answer"]
                st.write_stream(stream_data(resp))
            res_bot = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': resp}).execute()
            st.session_state.messages.append({"id": res_bot.data[0]['id'], "role": "assistant", "content": resp})

    # --- TAB 2: INSCRIPCIÓN ---
    with tab2:
        st.header(t["enroll_title"])
        
        @st.cache_data(ttl=60)
        def get_user_schedule(uid):
            regs = supabase.table('registrations').select('section_id').eq('user_id', uid).execute().data
            if not regs: return [], []
            s_ids = [r['section_id'] for r in regs]
            sch = supabase.table('sections').select('subject_id, day_of_week, start_time, end_time').in_('id', s_ids).execute().data
            return [{"day": s['day_of_week'], "start": dt_time.fromisoformat(s['start_time']), "end": dt_time.fromisoformat(s['end_time'])} for s in sch], [s['subject_id'] for s in sch]

        def check_conflict(schedule, new_sec):
            n_start, n_end = dt_time.fromisoformat(new_sec['start_time']), dt_time.fromisoformat(new_sec['end_time'])
            for s in schedule:
                if s['day'] == new_sec['day_of_week'] and max(s['start'], n_start) < min(s['end'], n_end): return True
            return False

        @st.cache_data(ttl=300)
        def get_all_subjects():
            return supabase.table('subjects').select('id, name, career, semester').order('name').execute().data

        subjects_data = get_all_subjects()

        if not subjects_data: 
            st.warning("No data.")
        else:
            # 1. Inicializar variables de estado
            if "selected_career" not in st.session_state:
                st.session_state.selected_career = t["filter_all"]
            if "selected_semester" not in st.session_state:
                st.session_state.selected_semester = t["filter_all_m"]

            # 2. Lógica de Filtrado Cruzado
            temp_data_car = subjects_data
            if st.session_state.selected_semester != t["filter_all_m"]:
                try:
                    sem_num_filter = int(st.session_state.selected_semester.split(" ")[1])
                    temp_data_car = [s for s in subjects_data if s['semester'] == sem_num_filter]
                except: pass
            
            avail_careers = sorted(list(set([s['career'] for s in temp_data_car if s['career']])))
            career_opts = [t["filter_all"]] + avail_careers

            temp_data_sem = subjects_data
            if st.session_state.selected_career != t["filter_all"]:
                temp_data_sem = [s for s in subjects_data if s['career'] == st.session_state.selected_career]
            
            avail_sems = sorted(list(set([s['semester'] for s in temp_data_sem if s['semester']])))
            semester_opts = [t["filter_all_m"]] + [f"Semestre {s}" for s in avail_sems]

            if st.session_state.selected_career not in career_opts:
                st.session_state.selected_career = t["filter_all"]
            if st.session_state.selected_semester not in semester_opts:
                st.session_state.selected_semester = t["filter_all_m"]

            # 3. Renderizar Selectbox con Keys
            c_f1, c_f2, c_res = st.columns([2, 2, 1])
            with c_f1:
                st.selectbox(t["filter_career"], career_opts, key="selected_career")
            with c_f2:
                st.selectbox(t["filter_sem"], semester_opts, key="selected_semester")
            
            def clear_filters_callback():
                st.session_state.selected_career = t["filter_all"]
                st.session_state.selected_semester = t["filter_all_m"]

            with c_res:
                st.write("")
                st.write("")
                st.button(t["reset_btn"], on_click=clear_filters_callback)

            filtered = subjects_data
            if st.session_state.selected_career != t["filter_all"]:
                filtered = [s for s in filtered if s['career'] == st.session_state.selected_career]
            if st.session_state.selected_semester != t["filter_all_m"]:
                try:
                    sem_num = int(st.session_state.selected_semester.split(" ")[1])
                    filtered = [s for s in filtered if s['semester'] == sem_num]
                except: pass

            s_dict = {s['name']: s['id'] for s in filtered}
            st.markdown(f"##### {t['search_label']}")
            sel_name = st.selectbox("Search", s_dict.keys(), index=None, placeholder=t["search_placeholder"], label_visibility="collapsed")
            
            st.divider()

            if sel_name:
                sid = s_dict[sel_name]
                secs = supabase.table('sections').select('*').eq('subject_id', sid).execute().data
                if not secs: st.warning("No sections.")
                else:
                    st.subheader(f"{t['sec_title']} {sel_name}")
                    sch, sids = get_user_schedule(user_id)
                    if sid in sids: st.info(t["msg_already"])
                    else:
                        for sec in secs:
                            with st.container(border=True):
                                rc = supabase.table('registrations').select('id', count='exact').eq('section_id', sec['id']).execute().count
                                cupos = sec['capacity'] - (rc if rc else 0)
                                c1,c2,c3,c4 = st.columns([2,3,2,2])
                                c1.write(f"**{sec['section_code']}**")
                                c2.write(f"{sec['day_of_week']} {sec['start_time'][:5]}-{sec['end_time'][:5]}")
                                c3.write(sec['professor_name'])
                                if cupos > 0:
                                    if c4.button(f"{t['btn_enroll']} ({cupos})", key=sec['id']):
                                        if check_conflict(sch, sec): st.error(t["msg_conflict"])
                                        else:
                                            supabase.table('registrations').insert({'user_id': user_id, 'section_id': sec['id']}).execute()
                                            st.success(t["msg_enrolled"])
                                            st.cache_data.clear()
                                            st.rerun()
                                else: c4.button(t["btn_full"], disabled=True, key=sec['id'])
        
        st.subheader(t["my_schedule"])
        sch, _ = get_user_schedule(user_id)
        if not sch: st.info(t["no_schedule"])
        else:
            regs = supabase.table('registrations').select('id, sections(section_code, day_of_week, start_time, end_time, professor_name, subjects(name))').eq('user_id', user_id).execute().data
            for r in regs:
                s = r['sections']
                with st.expander(f"📘 {s['subjects']['name']} ({s['section_code']})"):
                    c1,c2 = st.columns([4,1])
                    c1.write(f"{s['day_of_week']} {s['start_time'][:5]}-{s['end_time'][:5]} | Prof: {s['professor_name']}")
                    if c2.button(t["btn_drop"], key=f"del_{r['id']}", type="primary"):
                        supabase.table('registrations').delete().eq('id', r['id']).execute()
                        st.success(t["msg_dropped"])
                        st.cache_data.clear()
                        st.rerun()

    # --- TAB 3: ADMIN ---
    with tab3:
        st.header(t["admin_title"])
        admin_pass = st.text_input(t["admin_pass_label"], type="password")
        
        if admin_pass == ADMIN_PASSWORD:
            st.success(t["admin_success"])
            if st.button(t["admin_update_btn"], key="refresh_top"): st.rerun()
            
            try:
                response = supabase.table('chat_history').select('created_at, role, message, is_visible, user_id, feedback(rating, comment)').not_.is_('feedback', 'null').order('created_at', desc=True).execute()
                users_count = supabase.table('profiles').select('id', count='exact', head=True).execute().count

                if not response.data: 
                    st.warning("Aún no hay interacciones con feedback.")
                else:
                    # Métricas
                    total_feedback = len(response.data)
                    total_good = sum(1 for x in response.data if x['feedback'] and x['feedback'][0]['rating'] == 'good')
                    total_bad = sum(1 for x in response.data if x['feedback'] and x['feedback'][0]['rating'] == 'bad')
                    satisfaction_rate = (total_good / total_feedback * 100) if total_feedback > 0 else 0

                    st.markdown("### 📊 Métricas Generales")
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("👥 Usuarios Totales", users_count)
                    kpi2.metric("💬 Feedbacks Recibidos", total_feedback)
                    kpi3.metric("👍 Tasa Satisfacción", f"{satisfaction_rate:.1f}%")
                    kpi4.metric("🚨 Reportes Negativos", total_bad, delta_color="inverse")
                    
                    st.divider()

                    c_filter, _ = st.columns([0.4, 0.6])
                    show_only_bad = c_filter.toggle("🔥 Mostrar solo Feedback Negativo (Errores)", value=False)

                    data_tbl = []
                    st.write("Cargando detalles...")
                    pbar = st.progress(0)
                    
                    for i, item in enumerate(response.data):
                        fb = item['feedback'][0] if item['feedback'] else {'rating': 'N/A', 'comment': ''}
                        
                        if show_only_bad and fb['rating'] != 'bad':
                            pbar.progress((i+1)/len(response.data))
                            continue

                        icon = "✅" if fb['rating'] == "good" else "❌"
                        status = "Activo" if item['is_visible'] else "Archivado"
                        
                        try:
                            q = supabase.table('chat_history').select('message').eq('user_id', item['user_id']).eq('role', 'user').lt('created_at', item['created_at']).order('created_at', desc=True).limit(1).execute()
                            q_text = q.data[0]['message'] if q.data else "N/A"
                        except: q_text = "Error al buscar contexto"
                        
                        data_tbl.append({
                            t["col_date"]: item['created_at'][:16].replace("T", " "),
                            t["col_status"]: status,
                            t["col_q"]: q_text,
                            t["col_a"]: item['message'],
                            t["col_val"]: icon,
                            t["col_com"]: fb.get('comment', '')
                        })
                        pbar.progress((i+1)/len(response.data))
                    pbar.empty()
                    
                    st.markdown(f"### 📝 Registro Detallado ({len(data_tbl)} filas)")
                    
                    if data_tbl:
                        df_download = pd.DataFrame(data_tbl)
                        csv = df_download.to_csv(index=False).encode('utf-8-sig')
                        
                        c_tbl, c_dl = st.columns([0.8, 0.2])
                        c_tbl.dataframe(data_tbl, use_container_width=True)
                        c_dl.download_button(
                            label="📥 Descargar CSV",
                            data=csv,
                            file_name="reporte_chatbot.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("No hay datos para mostrar con los filtros actuales.")

            except Exception as e: st.error(f"Error cargando datos: {str(e)}")
        elif admin_pass: st.error(t["auth_error"])

# ==========================================
# LOGIN MANUAL
# ==========================================
else:
    col_L, col_Main, col_R = st.columns([1, 2, 1])
    with col_Main:
        st.subheader(t["login_title"])
        with st.form("login_form", enter_to_submit=False):
            input_email = st.text_input(t["login_user"])
            input_pass = st.text_input(t["login_pass"], type="password")
            submit = st.form_submit_button(t["login_btn"], use_container_width=True)
            if submit:
                all_users = fetch_all_users()
                if input_email in all_users:
                    stored_hash = all_users[input_email]['password_hash']
                    if bcrypt.checkpw(input_pass.encode('utf-8'), stored_hash.encode('utf-8')):
                        st.session_state["authentication_status"] = True
                        st.session_state["name"] = all_users[input_email]['full_name']
                        st.session_state["username"] = input_email
                        st.toast(t["login_welcome"])
                        time.sleep(0.5)
                        st.rerun()
                    else: st.error(t["login_failed"])
                else: st.error(t["login_failed"])

    with st.sidebar:
        st.subheader(t["reg_header"])
        with st.form("reg", enter_to_submit=False):
            n = st.text_input(t["reg_name"])
            e = st.text_input(t["reg_email"])
            p = st.text_input(t["reg_pass"], type="password")
            if st.form_submit_button(t["reg_btn"]):
                hashed_bytes = bcrypt.hashpw(p.encode('utf-8'), bcrypt.gensalt())
                hashed_str = hashed_bytes.decode('utf-8')
                try:
                    supabase.table('profiles').insert({'full_name': n, 'email': e, 'password_hash': hashed_str}).execute()
                    st.success(t["reg_success"])
                except: st.error("Error")