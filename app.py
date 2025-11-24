# Versión 11.1 (FINAL: Feedback en Tiempo Real - Muestra TODO)
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
    st.error("Error: Faltan claves de API en los Secrets de Streamlit.")
    st.stop()

# --- INICIALIZAR EL CLIENTE DE SUPABASE ---
@st.cache_resource
def init_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase_client()

# --- CACHING DE RECURSOS DEL CHATBOT ---
@st.cache_resource
def inicializar_cadena():
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

# --- AUTENTICACIÓN ---
def fetch_all_users():
    try:
        response = supabase.table('profiles').select("email, full_name, password_hash").execute()
        if not response.data: return {'usernames': {}}
        credentials = {'usernames': {}}
        for user in response.data:
            credentials['usernames'][user['email']] = {
                'email': user['email'], 'name': user['full_name'], 'password': user['password_hash']
            }
        return credentials
    except: return {'usernames': {}}

credentials = fetch_all_users()
authenticator = stauth.Authenticate(credentials, 'chatbot_duoc_cookie', 'abcdefg123456', cookie_expiry_days=30)

# --- INICIO DE LA APP ---
col_title1, col_title2 = st.columns([0.1, 0.9])
with col_title1: st.image(LOGO_ICON_URL, width=70)
with col_title2: st.title("Asistente Académico Duoc UC")

if st.session_state["authentication_status"] is True:
    user_name = st.session_state["name"]
    user_email = st.session_state["username"]
    
    st.sidebar.image(LOGO_BANNER_URL) 
    
    # Obtener ID de usuario
    if 'user_id' not in st.session_state:
        user_id_response = supabase.table('profiles').select('id').eq('email', user_email).execute()
        if user_id_response.data: st.session_state.user_id = user_id_response.data[0]['id']
        else: st.stop()
    user_id = st.session_state.user_id

    # Header
    c1, c2 = st.columns([0.8, 0.2])
    c1.caption(f"Conectado como: {user_name} ({user_email})")
    if c2.button("Cerrar Sesión", use_container_width=True):
        authenticator.logout()
        st.session_state.clear()
        st.rerun()

    # --- DEFINICIÓN DE PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs(["Chatbot de Reglamento", "Inscripción de Asignaturas", "🔒 Admin / Auditoría"])

    # --- PESTAÑA 1: CHATBOT ---
    with tab1:
        if st.button("Limpiar Historial del Chat", use_container_width=True, key="clear_chat"):
            with st.spinner("Archivando conversación..."):
                try:
                    # Soft Delete
                    supabase.table('chat_history').update({'is_visible': False}).eq('user_id', user_id).execute()
                    
                    st.session_state.messages = []
                    
                    welcome_msg = f"¡Hola {user_name}! Historial archivado. Los datos de feedback se han guardado."
                    res = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': welcome_msg}).execute()
                    
                    if res.data:
                        st.session_state.messages.append({"id": res.data[0]['id'], "role": "assistant", "content": welcome_msg})
                    
                    # Limpiar estados de formularios abiertos
                    keys_to_remove = [k for k in st.session_state.keys() if k.startswith("show_reason_")]
                    for k in keys_to_remove:
                        del st.session_state[k]

                    st.success("¡Chat limpio visualmente!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al limpiar: {e}")
        
        st.divider()
        retrieval_chain = inicializar_cadena()

        # Cargar Historial (VISIBLES)
        if "messages" not in st.session_state:
            st.session_state.messages = []
            history = supabase.table('chat_history').select('id, role, message').eq('user_id', user_id).eq('is_visible', True).order('created_at').execute()
            
            for row in history.data:
                st.session_state.messages.append({"id": row['id'], "role": row['role'], "content": row['message']})
            
            if not st.session_state.messages:
                welcome_msg = f"¡Hola {user_name}! Soy tu asistente del reglamento académico. ¿En qué te puedo ayudar hoy?"
                res = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': welcome_msg}).execute()
                if res.data:
                    st.session_state.messages.append({"id": res.data[0]['id'], "role": "assistant", "content": welcome_msg})

        # Mostrar Mensajes y Feedback con Comentarios
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Solo el asistente tiene feedback
                if msg["role"] == "assistant" and msg["id"]:
                    col_fb1, col_fb2, _ = st.columns([1,1,8])
                    
                    # --- BOTÓN LIKE ---
                    if col_fb1.button("👍", key=f"up_{msg['id']}"):
                        supabase.table('feedback').insert({
                            "message_id": msg['id'], 
                            "user_id": user_id, 
                            "rating": "good",
                            "comment": None 
                        }).execute()
                        st.toast("¡Gracias por tu valoración!")

                    # --- BOTÓN DISLIKE ---
                    reason_key = f"show_reason_{msg['id']}"
                    
                    if col_fb2.button("👎", key=f"down_{msg['id']}"):
                        st.session_state[reason_key] = True

                    if st.session_state.get(reason_key, False):
                        with st.form(key=f"form_{msg['id']}"):
                            st.write("Cuéntanos, ¿qué salió mal?")
                            comment_text = st.text_area("Comentario (Opcional):", placeholder="Ej: La respuesta es incorrecta...")
                            
                            col_sub1, col_sub2 = st.columns([1, 1])
                            with col_sub1:
                                if st.form_submit_button("Enviar Reporte"):
                                    supabase.table('feedback').insert({
                                        "message_id": msg['id'], 
                                        "user_id": user_id, 
                                        "rating": "bad",
                                        "comment": comment_text 
                                    }).execute()
                                    st.toast("Reporte enviado. ¡Gracias!")
                                    st.session_state[reason_key] = False 
                                    st.rerun()
                            with col_sub2:
                                if st.form_submit_button("Cancelar"):
                                    st.session_state[reason_key] = False 
                                    st.rerun()

        # Input Chat
        if prompt := st.chat_input("Escribe tu duda..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            supabase.table('chat_history').insert({'user_id': user_id, 'role': 'user', 'message': prompt}).execute()
            
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    resp = retrieval_chain.invoke({"input": prompt, "user_name": user_name})["answer"]
                    st.markdown(resp)
            
            res_bot = supabase.table('chat_history').insert({'user_id': user_id, 'role': 'assistant', 'message': resp}).execute()
            st.session_state.messages.append({"id": res_bot.data[0]['id'], "role": "assistant", "content": resp})
            st.rerun()

    # --- PESTAÑA 2: INSCRIPCIÓN ---
    with tab2:
        st.header("Inscripción de Asignaturas")

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
            st.warning("No hay datos.")
        else:
            # Filtros Cruzados
            current_career = st.session_state.get("filter_career", "Todas")
            current_semester_str = st.session_state.get("filter_semester", "Todos")

            if current_semester_str != "Todos":
                sem_num = int(current_semester_str.split(" ")[1])
                valid_careers_data = [s['career'] for s in subjects_data if s['semester'] == sem_num]
                unique_careers = sorted(list(set(valid_careers_data)))
            else:
                unique_careers = sorted(list(set([s['career'] for s in subjects_data if s['career']])))
            
            career_options = ["Todas"] + unique_careers

            if current_career != "Todas":
                valid_semesters_data = [s['semester'] for s in subjects_data if s['career'] == current_career]
                unique_semesters = sorted(list(set(valid_semesters_data)))
            else:
                unique_semesters = sorted(list(set([s['semester'] for s in subjects_data if s['semester']])))
            
            semester_options = ["Todos"] + [f"Semestre {s}" for s in unique_semesters]

            c_filter1, c_filter2, c_reset = st.columns([2, 2, 1])
            with c_filter1:
                try: idx_car = career_options.index(current_career)
                except ValueError: idx_car = 0 
                selected_career = st.selectbox("📂 Carrera:", options=career_options, index=idx_car, key="filter_career")
            with c_filter2:
                try: idx_sem = semester_options.index(current_semester_str)
                except ValueError: idx_sem = 0
                selected_semester = st.selectbox("⏳ Semestre:", options=semester_options, index=idx_sem, key="filter_semester")
            with c_reset:
                st.write("") 
                st.write("") 
                if st.button("🔄 Reset"):
                    del st.session_state["filter_career"]
                    del st.session_state["filter_semester"]
                    st.rerun()

            filtered_list = subjects_data
            if selected_career != "Todas":
                filtered_list = [s for s in filtered_list if s['career'] == selected_career]
            if selected_semester != "Todos":
                sem_val = int(selected_semester.split(" ")[1])
                filtered_list = [s for s in filtered_list if s['semester'] == sem_val]

            subjects_dict = {s['name']: s['id'] for s in filtered_list}

            st.markdown("##### 📚 Selecciona tu Ramo:")
            sel_subj_name = st.selectbox("Buscar:", options=subjects_dict.keys(), index=None, placeholder="Elige una asignatura...", label_visibility="collapsed")

            st.divider()

            if sel_subj_name:
                sid = subjects_dict[sel_subj_name]
                secs = supabase.table('sections').select('*').eq('subject_id', sid).execute().data
                
                if not secs: st.warning("No hay secciones abiertas.")
                else:
                    st.subheader(f"Secciones: {sel_subj_name}")
                    my_sch, my_sids = get_user_schedule(user_id)
                    if sid in my_sids: st.info("Ya tienes este ramo.")
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
                                    if c4.button(f"Inscribir ({cupos})", key=sec['id']):
                                        if check_conflict(my_sch, sec): st.error("Tope Horario")
                                        else:
                                            supabase.table('registrations').insert({'user_id': user_id, 'section_id': sec['id']}).execute()
                                            st.success("Inscrito!")
                                            st.cache_data.clear()
                                            st.rerun()
                                else: c4.button("Lleno", disabled=True, key=sec['id'])

        st.subheader("Tu Horario")
        sch, _ = get_user_schedule(user_id)
        if not sch: st.info("Sin inscripciones.")
        else:
            regs = supabase.table('registrations').select('id, sections(section_code, day_of_week, start_time, end_time, professor_name, subjects(name))').eq('user_id', user_id).execute().data
            for r in regs:
                s = r['sections']
                with st.expander(f"📘 {s['subjects']['name']} ({s['section_code']})"):
                    c1,c2 = st.columns([4,1])
                    c1.write(f"{s['day_of_week']} {s['start_time'][:5]}-{s['end_time'][:5]} | Prof: {s['professor_name']}")
                    if c2.button("Anular", key=f"del_{r['id']}", type="primary"):
                        supabase.table('registrations').delete().eq('id', r['id']).execute()
                        st.success("Eliminado")
                        st.cache_data.clear()
                        st.rerun()

    # --- PESTAÑA 3: AUDITORÍA (PROTEGIDA) ---
    with tab3:
        st.header("🕵️ Auditoría de Feedback (Zona Admin)")
        
        admin_pass = st.text_input("🔑 Ingrese Contraseña de Administrador:", type="password")
        
        if admin_pass == "DUOC2025":
            st.success("🔓 Acceso Concedido")
            st.info("Visualizando TODO el feedback recibido (Chats visibles y archivados).")

            if st.button("🔄 Actualizar Tabla"):
                st.rerun()

            try:
                # --- CORRECCIÓN AQUÍ: Quitamos el filtro .eq('is_visible', False) ---
                # Ahora mostramos TODO el feedback, sin importar si el chat está oculto o no
                response = supabase.table('chat_history')\
                    .select('created_at, role, message, is_visible, feedback(rating, comment)')\
                    .not_.is_('feedback', 'null')\
                    .order('created_at', desc=True)\
                    .execute()

                mensajes_con_feedback = response.data

                if not mensajes_con_feedback:
                    st.warning("No hay feedback registrado en la base de datos.")
                else:
                    data_para_tabla = []
                    for item in mensajes_con_feedback:
                        if item['feedback']:
                            fb = item['feedback'][0]
                            rating = fb['rating']
                            comment = fb.get('comment', '') 
                        else:
                            rating = "N/A"
                            comment = ""

                        icon = "✅ Positivo" if rating == "good" else "❌ Negativo"
                        estado_chat = "Activo" if item['is_visible'] else "Archivado"
                        
                        data_para_tabla.append({
                            "Fecha": item['created_at'][:16].replace("T", " "),
                            "Estado": estado_chat,
                            "Valoración": icon,
                            "Comentario": comment,
                            "Mensaje Bot": item['message']
                        })
                    
                    st.dataframe(data_para_tabla, use_container_width=True)

            except Exception as e:
                st.error(f"Error consultando base de datos: {e}")
        
        elif admin_pass:
            st.error("⛔ Contraseña Incorrecta")
        else:
            st.warning("⚠️ Esta zona es solo para administradores.")

else:
    authenticator.login(location='main')
    if st.session_state["authentication_status"] is False: st.error('Datos incorrectos')
    
    with st.sidebar:
        st.subheader("Registrarse")
        with st.form("reg"):
            n = st.text_input("Nombre")
            e = st.text_input("Email")
            p = st.text_input("Pass", type="password")
            if st.form_submit_button("Crear"):
                h = stauth.Hasher([p]).generate()[0]
                try:
                    supabase.table('profiles').insert({'full_name': n, 'email': e, 'password_hash': h}).execute()
                    st.success("Creado!")
                except Exception as err: st.error(f"Error: {err}")