import streamlit as st
import time

# --- CONFIGURACIÓN VISUAL ---
# Establecemos un título y layout, e inyectamos CSS básico para simular
# la estética que estás usando en la versión final.
st.set_page_config(page_title="Asistente Duoc UC v0.15 Beta", layout="wide", page_icon="🎓")
LOGO_ICON_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlve2kMlU53cq9Tl0DMxP0Ffo0JNap2dXq4q_uSdf4PyFZ9uraw7MU5irI6mA-HG8byNI&usqp=CAU"

st.markdown("""
    <style>
    /* Estilos CSS básicos embebidos para el prototipo */
    .main-header {font-size: 2rem; color: #003366; font-weight: 700;}
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { border-radius: 5px; background-color: #f0f2f6; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #003366 !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR SIMULADO (Login visual) ---
with st.sidebar:
    # Usamos el icono como logo simple para esta versión
    st.image(LOGO_ICON_URL, width=80)
    st.markdown("### Asistente Académico")
    st.caption("Versión Beta v0.15")
    st.divider()
    # Simulamos que un usuario ya ingresó
    st.info("👤 Usuario: alumno.demo@duocuc.cl")
    st.button("Cerrar Sesión (Simulado)")

# --- HEADER PRINCIPAL ---
col_h1, col_h2 = st.columns([0.1, 0.9])
with col_h1: st.image(LOGO_ICON_URL, width=60)
with col_h2: st.markdown("<div class='main-header'>Asistente Virtual Duoc UC</div>", unsafe_allow_html=True)

# --- ESTRUCTURA PRINCIPAL: PESTAÑAS ---
# Aquí se nota la gran diferencia con la v1.0
tab1, tab2 = st.tabs(["💬 Chat Reglamento", "📅 Toma de Ramos (Prototipo)"])

# --- TAB 1: CHATBOT (Simplificado, sin feedback ni chips) ---
with tab1:
    st.subheader("Consulta al Reglamento")
    st.caption("Haz preguntas sobre asistencia, notas o procesos académicos.")

    # Historial de sesión volátil (sin base de datos real)
    if "messages_v15" not in st.session_state:
        st.session_state.messages_v15 = [
            {"role": "assistant", "content": "¡Hola! Soy la versión beta del asistente. ¿Cuál es tu duda sobre el reglamento?"}
        ]

    # Mostrar historial
    for msg in st.session_state.messages_v15:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input del usuario
    if prompt := st.chat_input("Ej: ¿Cuánta asistencia necesito para aprobar?"):
        # 1. Mostrar mensaje usuario
        st.session_state.messages_v15.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # 2. Respuesta SIMULADA (para no depender de API keys en el screenshot)
        with st.chat_message("assistant"):
            with st.spinner("Consultando reglamento (simulación)..."):
                time.sleep(1.5) # Simular tiempo de espera
                # Respuesta genérica para la foto
                fake_response = """Basado en el Artículo 32 del reglamento académico (versión preliminar), la asistencia mínima requerida es del **75%** para asignaturas presenciales. 
                
*Nota: Esta es una respuesta simulada para la versión de prueba.*"""
                st.markdown(fake_response)
        st.session_state.messages_v15.append({"role": "assistant", "content": fake_response})

# --- TAB 2: TOMA DE RAMOS (Mockup Visual) ---
with tab2:
    st.subheader("Prototipo de Inscripción 2025")
    st.warning("🚧 Esta sección es un prototipo visual. Los datos no son reales y no conecta a base de datos.")

    # Filtros visuales (sin lógica real detrás)
    c_filt1, c_filt2, c_btn = st.columns([2, 2, 1])
    c_filt1.selectbox("📂 Carrera", ["Ingeniería en Informática", "Diseño Gráfico"], index=0)
    c_filt2.selectbox("⏳ Semestre", ["Semestre 1", "Semestre 3"], index=0)
    c_btn.write("")
    c_btn.write("")
    c_btn.button("Filtrar (Demo)")

    st.divider()
    st.markdown("##### Asignaturas Disponibles (Datos de Ejemplo)")

    # Ejemplos "hardcoded" para que se vea cómo será la interfaz
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
        c1.markdown("**📘 Programación de Algoritmos (PGY1121)**\nSec. 001D - Lunes 08:30")
        c2.write("Prof. Juan Pérez")
        c3.write("Cupos: **5** / 30")
        c4.button("Inscribir", key="btn_mock_1", type="primary")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
        c1.markdown("**📙 Base de Datos (BDD2130)**\nSec. 004N - Miércoles 19:00")
        c2.write("Prof. María López")
        c3.write("Cupos: **0** / 25")
        c4.button("Sin Cupos", key="btn_mock_2", disabled=True)

    st.divider()
    st.subheader("Tu Carga Académica")
    st.info("No tienes ramos inscritos en esta simulación.")