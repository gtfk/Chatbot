# Versión 27.0 (FINAL: Auto-Reparación de Perfil + Todo Integrado)
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
import time
from datetime import time as dt_time

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

load_css("styles.css")

# --- DICCIONARIO DE TRADUCCIONES ---
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
        "login_failed": "❌ Credenciales inválidas o correo no confirmado.",
        "login_welcome": "¡Bienvenido al Asistente!",
        "forgot_header": "¿Olvidaste tu contraseña?",
        "forgot_email": "Ingresa tu correo registrado",
        "forgot_btn": "Recuperar Contraseña",
        "forgot_success": "✅ Si el correo existe, te hemos enviado un enlace mágico.",
        "reset_title": "Restablecer Contraseña",
        "reset_pass_new": "Nueva Contraseña",
        "reset_btn_final": "Guardar Nueva Contraseña",
        "reset_success": "✅ Contraseña actualizada. Inicia sesión con tu nueva clave.",
        "chat_clear_btn": "🧹 Limpiar Conversación",
        "chat_cleaning": "Procesando solicitud...",
        "chat_cleaned": "¡Historial limpiado!",
        "chat_welcome": "¡Hola **{name}**! 👋 Soy tu asistente virtual de Duoc UC.",
        "chat_welcome_clean": "¡Hola **{name}**! El historial ha sido archivado.",
        "chat_placeholder": "Ej: ¿Con qué nota apruebo el ramo?",
        "chat_thinking": "Consultando reglamento...",
        "feedback_thanks": "¡Gracias por tu feedback! 👍",
        "feedback_report_sent": "Reporte enviado.",
        "feedback_modal_title": "¿Qué podemos mejorar?",
        "feedback_modal_placeholder": "Ej: La respuesta no es precisa...",
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
        "msg_enrolled": "✅ ¡Inscrito exitosamente!",
        "msg_conflict": "⛔ Error: Tope de Horario",
        "msg_already": "ℹ️ Ya estás inscrito.",
        "my_schedule": "Tu Carga Académica",
        "no_schedule": "No tienes ramos inscritos.",
        "btn_drop": "Anular Ramo",
        "msg_dropped": "Asignatura eliminada.",
        "admin_title": "Panel de Control (Admin)",
        "admin_pass_label": "Clave de Acceso:",
        "admin_success": "Acceso Autorizado",
        "admin_info": "Registro de auditoría.",
        "admin_update_btn": "🔄 Refrescar Datos",
        "col_date": "Fecha",
        "col_status": "Estado",
        "col_q": "Pregunta",
        "col_a": "Respuesta",
        "col_val": "Eval",
        "col_com": "Detalle",
        "reg_header": "Crear Cuenta Alumno",
        "reg_name": "Nombre y Apellido",
        "reg_email": "Correo Duoc",
        "reg_pass": "Crear Contraseña",
        "reg_btn": "Registrarse",
        "reg_success": "¡Cuenta creada! Revisa tu correo para confirmar.",
        "auth_error": "Verifica tus datos.",
        "system_prompt": "INSTRUCCIÓN: Responde en Español formal pero cercano. ROL: Coordinador académico Duoc UC."
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
        "login_failed": "❌ Invalid credentials or email not confirmed.",
        "login_welcome": "Welcome to the Assistant!",
        "forgot_header": "Forgot password?",
        "forgot_email": "Enter registered email",
        "forgot_btn": "Recover Password",
        "forgot_success": "✅ If email exists, a magic link has been sent.",
        "reset_title": "Reset Password",
        "reset_pass_new": "New Password",
        "reset_btn_final": "Save New Password",
        "reset_success": "✅ Password updated. Login with your new password.",
        "chat_clear_btn": "🧹 Clear Conversation",
        "chat_cleaning": "Processing...",
        "chat_cleaned": "History cleared!",
        "chat_welcome": "Hello **{name}**! 👋 I'm your Duoc UC virtual assistant.",
        "chat_welcome_clean": "Hello **{name}**! History archived.",
        "chat_placeholder": "Ex: What is the passing grade?",
        "chat_thinking": "Consulting rulebook...",
        "feedback_thanks": "Thanks! 👍",
        "feedback_report_sent": "Report sent.",
        "feedback_modal_title": "What went wrong?",
        "feedback_modal_placeholder": "Ex: Inaccurate info...",
        "btn_send": "Send Comment",
        "btn_cancel": "Skip",
        "enroll_title": "Course Registration 2025",
        "filter_career": "📂 Career:",
        "filter_sem": "⏳ Semester:",
        "filter_all": "All Careers",
        "filter_all_m": "All Semesters",
        "reset_btn": "🔄 Clear Filters",
        "search_label": "📚 Search Subject:",
        "search_placeholder": "Type subject name...",
        "sec_title": "Available Sections for:",
        "btn_enroll": "Enroll",
        "btn_full": "Full",
        "msg_enrolled": "✅ Enrolled successfully!",
        "msg_conflict": "⛔ Error: Schedule Conflict",
        "msg_already": "ℹ️ Already enrolled.",
        "my_schedule": "Your Academic Load",
        "no_schedule": "No subjects enrolled.",
        "btn_drop": "Drop",
        "msg_dropped": "Subject removed.",
        "admin_title": "Control Panel (Admin)",
        "admin_pass_label": "Access Key:",
        "admin_success": "Access Granted",
        "admin_info": "Audit log.",
        "admin_update_btn": "🔄 Refresh",
        "col_date": "Date",
        "col_status": "Status",
        "col_q": "Question",
        "col_a": "Answer",
        "col_val": "Rate",
        "col_com": "Detail",
        "reg_header": "Create Account",
        "reg_name": "Full Name",
        "reg_email": "Duoc Email",
        "reg_pass": "Create Password",
        "reg_btn": "Register",
        "reg_success": "Account created! Please check email to confirm.",
        "auth_error": "Check credentials.",
        "system_prompt": "INSTRUCTION: Respond in English. ROLE: Academic coordinator Duoc UC."
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
c1, c2 = st.columns([0.1, 0.9])
with c1: st.image(LOGO_ICON_URL, width=70)
with c2: st.title(t["title"])

# --- AUTO-LOGIN & RECOVERY CHECK ---
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

try:
    # Verificar sesión activa o enlace mágico
    session = supabase.auth.get_session()
    if session:
        # 1. MODO RESET PASSWORD: Si el usuario entró por link de recuperación pero la app no lo sabe
        # Detectamos si estamos en un flujo de recuperación (url params o estado interno)
        # Por simplicidad, si hay session pero no authentication_status, asumimos login o recovery exitoso.
        
        st.session_state["authentication_status"] = True
        st.session_state["user_id"] = session.user.id
        st.session_state["username"] = session.user.email
        
        # --- AUTO-REPARACIÓN DE PERFIL (CRÍTICO PARA TU ERROR) ---
        # Verificamos si existe en la tabla profiles
        try:
            prof = supabase.table('profiles').select('full_name').eq('id', session.user.id).execute()
            if not prof.data:
                # NO EXISTE: LO CREAMOS AHORA MISMO
                supabase.table('profiles').insert({
                    'id': session.user.id,
                    'email': session.user.email,
                    'full_name': session.user.user_metadata.get('full_name', 'Estudiante')
                }).execute()
                st.session_state["name"] = session.user.user_metadata.get('full_name', 'Estudiante')
            else:
                st.session_state["name"] = prof.data[0]['full_name']
        except Exception as e:
            st.error(f"Error sincronizando perfil: {e}")
            
except: pass


# ==========================================
# APP PRINCIPAL (LOGUEADO)
# ==========================================
if st.session_state["authentication_status"] is True:
    user_name = st.session_state["name"]
    user_id = st.session_state["user_id"]

    c1, c2 = st.columns([0.8, 0.2])
    c1.caption(f"{t['login_success']} {user_name}")
    if c2.button(t["logout_btn"], use_container_width=True):
        try:
            supabase.auth.sign_out() 
        except: pass
        st.session_state.clear()
        st.rerun()

    tab1, tab2, tab3 = st.tabs([t["tab1"], t["tab2"], t["tab3"]])

    # --- TAB 1: CHATBOT ---
    with tab1:
        if st.button(t["chat_clear_btn"], use_container_width=True):
            supabase.table('chat_history').update({'is_visible': False}).eq('user_id', user_id).execute()
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        chain = inicializar_cadena(lang_code)

        if "messages" not in st.session_state:
            st.session_state.messages = []
            history = supabase.table('chat_history').select('*').eq('user_id', user_id).eq('is_visible', True).order('created_at').execute()
            for row in history.data:
                st.session_state.messages.append({"id": row['id'], "role": row['role'], "content": row['message']})
            if not st.session_state.messages:
                msg = t["chat_welcome"].format(name=user_name)
                res = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': msg}).execute()
                st.session_state.messages.append({"id": res.data[0]['id'], "role": "assistant", "content": msg})

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("id"):
                    c1, c2, _ = st.columns([1,1,10])
                    if c1.button("👍", key=f"up_{msg['id']}"):
                        supabase.table('feedback').insert({"message_id": msg['id'], "user_id": user_id, "rating": "good"}).execute()
                        st.toast(t["feedback_thanks"])
                    if c2.button("👎", key=f"down_{msg['id']}"):
                        st.session_state[f"show_reason_{msg['id']}"] = True
                    
                    if st.session_state.get(f"show_reason_{msg['id']}", False):
                        with st.form(key=f"f_{msg['id']}", enter_to_submit=False):
                            comment = st.text_area(t["feedback_modal_placeholder"])
                            if st.form_submit_button(t["btn_send"]):
                                supabase.table('feedback').insert({"message_id": msg['id'], "user_id": user_id, "rating": "bad", "comment": comment}).execute()
                                st.toast("OK")
                                st.session_state[f"show_reason_{msg['id']}"] = False
                                st.rerun()

        if prompt := st.chat_input(t["chat_placeholder"]):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            supabase.table('chat_history').insert({'user_id': user_id, 'role': 'user', 'message': prompt}).execute()
            
            with st.chat_message("assistant"):
                with st.spinner(t["chat_thinking"]):
                    resp = chain.invoke({"input": prompt, "user_name": user_name})["answer"]
                st.write_stream(stream_data(resp))
            
            r = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': resp}).execute()
            st.session_state.messages.append({"id": r.data[0]['id'], "role": "assistant", "content": resp})

    # TAB 2: INSCRIPCION
    with tab2:
        st.header(t["enroll_title"])
        # Fetch subjects
        subs = supabase.table('subjects').select('*').order('name').execute().data
        if subs:
            cars = sorted(list(set([s['career'] for s in subs])))
            sems = sorted(list(set([s['semester'] for s in subs])))
            
            c1, c2, c3 = st.columns([2,2,1])
            sel_car = c1.selectbox(t["filter_career"], [t["filter_all"]] + cars)
            sel_sem = c2.selectbox(t["filter_sem"], [t["filter_all_m"]] + [f"Sem {x}" for x in sems])
            if c3.button(t["reset_btn"]): st.rerun()

            filt = subs
            if sel_car != t["filter_all"]: filt = [s for s in filt if s['career'] == sel_car]
            if sel_sem != t["filter_all_m"]: filt = [s for s in filt if s['semester'] == int(sel_sem.split()[1])]
            
            sub_map = {s['name']: s['id'] for s in filt}
            target = st.selectbox(t["search_label"], sub_map.keys(), index=None)
            
            if target:
                secs = supabase.table('sections').select('*').eq('subject_id', sub_map[target]).execute().data
                my_regs = [r['section_id'] for r in supabase.table('registrations').select('section_id').eq('user_id', user_id).execute().data]
                my_sch = supabase.table('sections').select('*').in_('id', my_regs).execute().data
                
                if sub_map[target] in [s['subject_id'] for s in my_sch]:
                    st.info(t["msg_already"])
                else:
                    for sec in secs:
                        with st.container(border=True):
                            cnt = supabase.table('registrations').select('id', count='exact').eq('section_id', sec['id']).execute().count
                            cap = sec['capacity'] - (cnt if cnt else 0)
                            cc1, cc2, cc3 = st.columns([3,2,2])
                            cc1.write(f"**{sec['section_code']}**")
                            cc1.caption(sec['professor_name'])
                            cc2.write(f"{sec['day_of_week']}")
                            cc2.caption(f"{sec['start_time'][:5]} - {sec['end_time'][:5]}")
                            
                            if cap > 0:
                                if cc3.button(f"{t['btn_enroll']} ({cap})", key=sec['id']):
                                    # Conflict check
                                    conflict = False
                                    n_s, n_e = dt_time.fromisoformat(sec['start_time']), dt_time.fromisoformat(sec['end_time'])
                                    for m in my_sch:
                                        m_s, m_e = dt_time.fromisoformat(m['start_time']), dt_time.fromisoformat(m['end_time'])
                                        if m['day_of_week'] == sec['day_of_week'] and max(m_s, n_s) < min(m_e, n_e):
                                            conflict = True
                                    
                                    if conflict: st.error(t["msg_conflict"])
                                    else:
                                        supabase.table('registrations').insert({'user_id': user_id, 'section_id': sec['id']}).execute()
                                        st.success(t["msg_enrolled"])
                                        st.rerun()
                            else: cc3.button("Full", disabled=True)

        # Schedule
        st.divider()
        st.subheader(t["my_schedule"])
        my_regs_data = supabase.table('registrations').select('id, sections(subject_id, section_code, day_of_week, start_time, end_time, professor_name, subjects(name))').eq('user_id', user_id).execute().data
        if not my_regs_data: st.info(t["no_schedule"])
        else:
            for r in my_regs_data:
                s = r['sections']
                with st.expander(f"📘 {s['subjects']['name']}"):
                    st.write(f"**{s['section_code']}** | {s['day_of_week']} {s['start_time'][:5]}-{s['end_time'][:5]}")
                    if st.button(t["btn_drop"], key=f"d_{r['id']}"):
                        supabase.table('registrations').delete().eq('id', r['id']).execute()
                        st.rerun()

    # TAB 3: ADMIN
    with tab3:
        st.header(t["admin_title"])
        adm_p = st.text_input(t["admin_pass_label"], type="password")
        if adm_p == ADMIN_PASSWORD:
            if st.button(t["admin_update_btn"]): st.rerun()
            # Fetch audit
            audit = supabase.table('chat_history').select('created_at, role, message, is_visible, user_id, feedback(rating, comment)').not_.is_('feedback', 'null').order('created_at', desc=True).execute()
            if audit.data:
                clean_data = []
                for row in audit.data:
                    fb = row['feedback'][0] if row['feedback'] else {}
                    clean_data.append({
                        t["col_date"]: row['created_at'][:16].replace("T"," "),
                        t["col_status"]: "Active" if row['is_visible'] else "Archived",
                        t["col_a"]: row['message'],
                        t["col_val"]: "✅" if fb.get('rating')=='good' else "❌",
                        t["col_com"]: fb.get('comment', '')
                    })
                st.dataframe(clean_data, use_container_width=True)
            else: st.info("No data")
        
    # --- BARRA LATERAL EXTRA (CAMBIAR CLAVE) ---
    # Solo visible si está logueado
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔐 Seguridad")
        with st.expander("Cambiar Contraseña"):
            with st.form("pass_change", enter_to_submit=False):
                new_p = st.text_input("Nueva Contraseña", type="password")
                if st.form_submit_button("Actualizar"):
                    if len(new_p) >= 6:
                        try:
                            supabase.auth.update_user({"password": new_p})
                            st.success("✅ Contraseña actualizada")
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else: st.error("Mínimo 6 caracteres")

# ==========================================
# PANTALLA DE LOGIN (NO LOGUEADO)
# ==========================================
else:
    
    # --- DETECCIÓN DE LINK MÁGICO (RECUPERACIÓN) ---
    # Si supabase.auth.get_session() retorna algo al inicio, el código de arriba (línea 124)
    # ya habrá seteado authentication_status=True y redirigido.
    # Si estamos aquí, es porque NO hay sesión válida.

    cL, cM, cR = st.columns([1, 2, 1])
    with cM:
        st.subheader(t["login_title"])
        with st.form("log", enter_to_submit=False):
            e = st.text_input(t["login_user"])
            p = st.text_input(t["login_pass"], type="password")
            if st.form_submit_button(t["login_btn"], use_container_width=True):
                try:
                    res = supabase.auth.sign_in_with_password({"email": e, "password": p})
                    st.session_state["authentication_status"] = True
                    st.rerun()
                except: st.error(t["login_failed"])
        
        with st.expander(t["forgot_header"]):
            with st.form("rec", enter_to_submit=False):
                rec_e = st.text_input(t["forgot_email"])
                if st.form_submit_button(t["forgot_btn"]):
                    try:
                        # Ajusta la URL a tu dominio real de Streamlit
                        supabase.auth.reset_password_for_email(rec_e, options={'redirect_to': 'https://chatbot-duoc1.streamlit.app'})
                        st.success(t["forgot_success"])
                    except: st.error("Error")

    with st.sidebar:
        st.subheader(t["reg_header"])
        with st.form("signin", enter_to_submit=False):
            rn = st.text_input(t["reg_name"])
            re = st.text_input(t["reg_email"])
            rp = st.text_input(t["reg_pass"], type="password")
            if st.form_submit_button(t["reg_btn"]):
                try:
                    # Crear en Auth
                    res = supabase.auth.sign_up({"email": re, "password": rp, "options": {"data": {"full_name": rn}}})
                    # Crear Perfil (Fix APIError)
                    if res.user:
                        supabase.table('profiles').insert({'id': res.user.id, 'email': re, 'full_name': rn}).execute()
                        st.success(t["reg_success"])
                    else:
                        st.info("Usuario existente o requiere confirmación.")
                except Exception as ex: st.error(str(ex))