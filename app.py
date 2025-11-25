# Versión 32.0 (FINAL: Fix Rerun Exception + OTP Robusto + Todo Integrado)
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

# --- CARGAR CSS ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"⚠️ Falta {file_name}.")

load_css("styles.css")

# --- DICCIONARIO DE TRADUCCIONES ---
TEXTS = {
    "es": {
        "label": "Español 🇨🇱",
        "title": "Asistente Académico Duoc UC",
        "login_success": "Usuario:",
        "logout_btn": "Cerrar Sesión",
        "tab1": "💬 Chatbot Reglamento",
        "tab2": "📅 Inscripción de Asignaturas",
        "tab3": "🔐 Admin / Auditoría",
        "login_title": "Iniciar Sesión",
        "login_user": "Correo Institucional",
        "login_pass": "Contraseña",
        "login_btn": "Ingresar",
        "login_failed": "❌ Credenciales inválidas.",
        "login_welcome": "¡Bienvenido al Asistente!",
        "forgot_header": "¿Olvidaste tu contraseña?",
        "forgot_email": "Ingresa tu correo registrado",
        "forgot_step1_btn": "Enviar Código de Acceso",
        "forgot_step1_msg": "Te enviaremos un código de 6 dígitos a tu correo.",
        "forgot_step2_label": "Ingresa el Código (OTP)",
        "forgot_step2_btn": "Verificar y Entrar",
        "forgot_success": "✅ Código verificado. Ahora cambia tu contraseña en el menú lateral.",
        "forgot_error_otp": "❌ Código inválido o expirado.",
        "change_pass_header": "Cambiar Contraseña",
        "new_pass": "Nueva Contraseña",
        "change_pass_btn": "Actualizar Clave",
        "pass_updated": "✅ Contraseña actualizada.",
        "chat_clear_btn": "🧹 Limpiar Conversación",
        "chat_cleaned": "¡Historial limpiado!",
        "chat_welcome": "¡Hola **{name}**! 👋 Soy tu asistente virtual.",
        "feedback_thanks": "¡Gracias! 👍",
        "feedback_report_sent": "Reporte enviado.",
        "feedback_modal_title": "¿Qué podemos mejorar?",
        "feedback_modal_placeholder": "Ej: La respuesta no es precisa...",
        "btn_send": "Enviar",
        "btn_cancel": "Omitir",
        "enroll_title": "Toma de Ramos 2025",
        "filter_career": "📂 Filtrar por Carrera:",
        "filter_sem": "⏳ Filtrar por Semestre:",
        "filter_all": "Todas",
        "filter_all_m": "Todos",
        "reset_btn": "🔄 Limpiar",
        "search_label": "📚 Buscar:",
        "btn_enroll": "Inscribir",
        "msg_enrolled": "✅ ¡Inscrito!",
        "msg_conflict": "⛔ Tope de Horario",
        "msg_already": "ℹ️ Ya inscrito.",
        "my_schedule": "Tu Carga",
        "no_schedule": "Sin ramos.",
        "btn_drop": "Anular",
        "admin_title": "Panel Admin",
        "admin_pass_label": "Clave Admin:",
        "reg_header": "Crear Cuenta",
        "reg_name": "Nombre",
        "reg_email": "Correo",
        "reg_pass": "Clave",
        "reg_btn": "Registrarse",
        "reg_success": "¡Cuenta creada!",
        "chat_placeholder": "Escribe tu duda...",
        "chat_thinking": "Pensando...",
        "system_prompt": "INSTRUCCIÓN: Responde en Español formal pero cercano. ROL: Coordinador Duoc UC."
    },
    "en": {
        "label": "English 🇺🇸",
        "title": "Duoc UC Assistant",
        "login_success": "User:",
        "logout_btn": "Log Out",
        "tab1": "💬 Chat",
        "tab2": "📅 Enrollment",
        "tab3": "🔐 Admin",
        "login_title": "Student Login",
        "login_user": "Email",
        "login_pass": "Password",
        "login_btn": "Login",
        "login_failed": "❌ Invalid credentials.",
        "login_welcome": "Welcome!",
        "forgot_header": "Forgot password?",
        "forgot_email": "Enter registered email",
        "forgot_step1_btn": "Send Access Code",
        "forgot_step1_msg": "We will send a 6-digit code to your email.",
        "forgot_step2_label": "Enter Code (OTP)",
        "forgot_step2_btn": "Verify & Login",
        "forgot_success": "✅ Code verified. Please change password in sidebar.",
        "forgot_error_otp": "❌ Invalid code.",
        "change_pass_header": "Change Password",
        "new_pass": "New Password",
        "change_pass_btn": "Update Password",
        "pass_updated": "✅ Password updated.",
        "chat_clear_btn": "🧹 Clear Chat",
        "chat_cleaned": "Cleared!",
        "chat_welcome": "Hello **{name}**! 👋",
        "feedback_thanks": "Thanks! 👍",
        "feedback_report_sent": "Report sent.",
        "feedback_modal_title": "What's wrong?",
        "feedback_modal_placeholder": "Ex: Inaccurate info...",
        "btn_send": "Send",
        "btn_cancel": "Skip",
        "enroll_title": "Enrollment 2025",
        "filter_career": "📂 Career:",
        "filter_sem": "⏳ Semester:",
        "filter_all": "All",
        "filter_all_m": "All",
        "reset_btn": "🔄 Reset",
        "search_label": "📚 Search:",
        "btn_enroll": "Enroll",
        "msg_enrolled": "✅ Enrolled!",
        "msg_conflict": "⛔ Conflict",
        "msg_already": "ℹ️ Joined.",
        "my_schedule": "Your Load",
        "no_schedule": "Empty.",
        "btn_drop": "Drop",
        "admin_title": "Admin Panel",
        "admin_pass_label": "Admin Key:",
        "reg_header": "Sign Up",
        "reg_name": "Name",
        "reg_email": "Email",
        "reg_pass": "Password",
        "reg_btn": "Register",
        "reg_success": "Created!",
        "chat_placeholder": "Ask here...",
        "chat_thinking": "Thinking...",
        "system_prompt": "INSTRUCTION: Respond in English. ROLE: Academic coordinator Duoc UC."
    }
}

# --- API KEYS ---
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

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

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
    prompt_template = TEXTS[language_code]["system_prompt"] + """
    CONTEXT: {context}
    QUESTION: {input}
    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"""<div class="sidebar-logo-container"><img src="{LOGO_BANNER_URL}" style="width:100%;max-width:180px;"></div>""", unsafe_allow_html=True)
    lang_option = st.selectbox("🌐 Language / Idioma", ["Español 🇨🇱", "English 🇺🇸"], format_func=lambda x: TEXTS["es" if "Español" in x else "en"]["label"])
    lang_code = "es" if "Español" in lang_option else "en"
    t = TEXTS[lang_code]

# --- CABECERA ---
c1, c2 = st.columns([0.1, 0.9])
with c1: st.image(LOGO_ICON_URL, width=70)
with c2: st.title(t["title"])

# --- AUTO-LOGIN ---
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

try:
    session = supabase.auth.get_session()
    if session and not st.session_state["authentication_status"]:
        st.session_state["authentication_status"] = True
        st.session_state["user_id"] = session.user.id
        st.session_state["username"] = session.user.email
        try:
            prof = supabase.table('profiles').select('full_name').eq('id', session.user.id).execute()
            if prof.data: st.session_state["name"] = prof.data[0]['full_name']
            else:
                n = session.user.user_metadata.get('full_name', 'Estudiante')
                supabase.table('profiles').upsert({'id': session.user.id, 'email': session.user.email, 'full_name': n}).execute()
                st.session_state["name"] = n
        except: st.session_state["name"] = "Estudiante"
        st.rerun()
except: pass

# ==========================================
# APP PRINCIPAL (LOGUEADO)
# ==========================================
if st.session_state["authentication_status"] is True:
    user_name = st.session_state["name"]
    user_id = st.session_state["user_id"]
    user_email = st.session_state["username"]

    c1, c2 = st.columns([0.8, 0.2])
    c1.caption(f"{t['login_success']} {user_name}")
    if c2.button(t["logout_btn"], use_container_width=True):
        try:
            supabase.auth.sign_out() 
        except: pass
        st.session_state.clear()
        st.rerun()

    with st.sidebar:
        st.markdown("---")
        with st.expander(t["change_pass_header"]):
            with st.form("pass_change", enter_to_submit=False):
                new_p = st.text_input(t["new_pass"], type="password")
                if st.form_submit_button(t["change_pass_btn"]):
                    if len(new_p) >= 6:
                        try:
                            supabase.auth.update_user({"password": new_p})
                            st.success(t["pass_updated"])
                        except Exception as e: st.error(f"Error: {e}")
                    else: st.error("Min 6 chars")

    tab1, tab2, tab3 = st.tabs([t["tab1"], t["tab2"], t["tab3"]])

    # --- TAB 1: CHAT ---
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
            for r in history.data:
                st.session_state.messages.append({"id": r['id'], "role": r['role'], "content": r['message']})
            if not st.session_state.messages:
                msg = t["chat_welcome"].format(name=user_name)
                try:
                    res = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': msg}).execute()
                    st.session_state.messages.append({"id": res.data[0]['id'], "role": "assistant", "content": msg})
                except Exception as e:
                    try:
                        supabase.table('profiles').upsert({'id': user_id, 'email': user_email, 'full_name': user_name}).execute()
                        res = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': msg}).execute()
                        st.session_state.messages.append({"id": res.data[0]['id'], "role": "assistant", "content": msg})
                    except: st.error("Error DB")

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
            
            try:
                supabase.table('chat_history').insert({'user_id': user_id, 'role': 'user', 'message': prompt}).execute()
            except:
                supabase.table('profiles').upsert({'id': user_id, 'email': user_email, 'full_name': user_name}).execute()
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
                
                if sub_map[target] in [s['subject_id'] for s in my_sch]: st.info(t["msg_already"])
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
                                    conflict = False
                                    n_s, n_e = dt_time.fromisoformat(sec['start_time']), dt_time.fromisoformat(sec['end_time'])
                                    for m in my_sch:
                                        m_s, m_e = dt_time.fromisoformat(m['start_time']), dt_time.fromisoformat(m['end_time'])
                                        if m['day_of_week'] == sec['day_of_week'] and max(m_s, n_s) < min(m_e, n_e): conflict = True
                                    if conflict: st.error(t["msg_conflict"])
                                    else:
                                        supabase.table('registrations').insert({'user_id': user_id, 'section_id': sec['id']}).execute()
                                        st.success(t["msg_enrolled"])
                                        st.rerun()
                            else: cc3.button("Full", disabled=True)
        
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
            audit = supabase.table('chat_history').select('created_at, role, message, is_visible, user_id, feedback(rating, comment)').not_.is_('feedback', 'null').order('created_at', desc=True).execute()
            if audit.data:
                clean_data = []
                for row in audit.data:
                    fb = row['feedback'][0] if row['feedback'] else {}
                    try:
                        q = supabase.table('chat_history').select('message').eq('user_id', row['user_id']).eq('role', 'user').lt('created_at', row['created_at']).order('created_at', desc=True).limit(1).execute()
                        q_text = q.data[0]['message'] if q.data else "N/A"
                    except: q_text = "Error"
                    clean_data.append({
                        t["col_date"]: row['created_at'][:16].replace("T"," "),
                        t["col_status"]: "Active" if row['is_visible'] else "Archived",
                        t["col_q"]: q_text,
                        t["col_a"]: row['message'],
                        t["col_val"]: "✅" if fb.get('rating')=='good' else "❌",
                        t["col_com"]: fb.get('comment', '')
                    })
                st.dataframe(clean_data, use_container_width=True)
            else: st.info("No data")

else:
    cL, cM, cR = st.columns([1, 2, 1])
    with cM:
        st.subheader(t["login_title"])
        with st.form("log", enter_to_submit=False):
            e = st.text_input(t["login_user"])
            p = st.text_input(t["login_pass"], type="password")
            if st.form_submit_button(t["login_btn"], use_container_width=True):
                try:
                    res = supabase.auth.sign_in_with_password({"email": e, "password": p})
                    st.rerun()
                except Exception as e: 
                    st.error(f"Error de Login: {e}")
        
        # --- RECUPERACIÓN OTP ---
        with st.expander(t["forgot_header"]):
            if "recovery_mode" not in st.session_state:
                with st.form("send_otp_form", enter_to_submit=False):
                    rec_e = st.text_input(t["forgot_email"])
                    if st.form_submit_button(t["forgot_step1_btn"]):
                        if rec_e:
                            try:
                                supabase.auth.sign_in_with_otp({"email": rec_e})
                                st.session_state["recovery_mode"] = True
                                st.session_state["recovery_email"] = rec_e
                                st.rerun()
                            except Exception as e: st.error(f"Error: {e}")
                        else: st.warning("Ingresa un correo.")
            else:
                st.info(t["forgot_step1_msg"])
                with st.form("verify_otp_form", enter_to_submit=False):
                    otp_code = st.text_input(t["forgot_step2_label"])
                    if st.form_submit_button(t["forgot_step2_btn"]):
                        try:
                            # === CORRECCIÓN CLAVE: except Exception as e para atrapar error y dejar pasar rerun ===
                            res = supabase.auth.verify_otp({"email": st.session_state["recovery_email"], "token": otp_code, "type": "email"})
                            if res.user:
                                st.session_state["authentication_status"] = True
                                st.session_state["user_id"] = res.user.id
                                st.session_state["username"] = res.user.email
                                st.success(t["forgot_success"])
                                time.sleep(2)
                                st.rerun()
                        except Exception as e: 
                             # Filtramos la excepción de rerun para no mostrar error si fue éxito
                             if "RerunData" not in str(type(e)):
                                 st.error(f"{t['forgot_error_otp']}")

    with st.sidebar:
        st.subheader(t["reg_header"])
        with st.form("signin", enter_to_submit=False):
            rn = st.text_input(t["reg_name"])
            re = st.text_input(t["reg_email"])
            rp = st.text_input(t["reg_pass"], type="password")
            if st.form_submit_button(t["reg_btn"]):
                try:
                    res = supabase.auth.sign_up({"email": re, "password": rp, "options": {"data": {"full_name": rn}}})
                    if res.user:
                        supabase.table('profiles').upsert({'id': res.user.id, 'email': re, 'full_name': rn}).execute()
                        st.success(t["reg_success"])
                    else:
                        st.info("Check email")
                except Exception as ex: st.error(str(ex))