import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Duoc UC Bot v0.8 (Admin)", layout="wide", page_icon="🔐")

# --- ESTILOS SIMULADOS ---
st.markdown("""
    <style>
    .main-header {font-size: 1.8rem; color: #003366; font-weight: bold;}
    .metric-card {background-color: #f9f9f9; padding: 15px; border-radius: 8px; border: 1px solid #ddd;}
    </style>
""", unsafe_allow_html=True)

# --- DATOS MOCK (FALSOS) PARA EL PANEL ADMIN ---
def get_mock_logs():
    # Generamos datos que parezcan reales para la tabla
    data = []
    actions = ["Login", "Consulta Reglamento", "Intento Fallido", "Inscripción", "Feedback Negativo"]
    users = ["j.perez@duocuc.cl", "m.gonzalez@duocuc.cl", "admin@duoc.cl", "a.rojas@duocuc.cl"]
    
    for i in range(10):
        t = datetime.now() - timedelta(minutes=random.randint(1, 300))
        data.append({
            "Timestamp": t.strftime("%Y-%m-%d %H:%M"),
            "Usuario": random.choice(users),
            "Acción": random.choice(actions),
            "Latencia (ms)": random.randint(50, 1200),
            "Estado": random.choice(["✅ OK", "✅ OK", "⚠️ Alerta"])
        })
    return pd.DataFrame(data)

# --- SIDEBAR: SIMULACIÓN DE REGISTRO Y LOGIN ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Logo_DuocUC.svg/2560px-Logo_DuocUC.svg.png", width=150)
    st.caption("v0.8.1 - Módulo Administrativo")
    
    # Selector para la demo (Para que puedas sacar pantallazos de ambos estados)
    app_mode = st.radio("Modo de Vista (Para Screenshot)", ["Vista: Registro (Logout)", "Vista: Admin (Login)"])

    st.divider()

    if app_mode == "Vista: Registro (Logout)":
        st.subheader("🔐 Acceso Estudiantes")
        tab_login, tab_reg = st.tabs(["Ingresar", "Registrarse"])
        
        with tab_login:
            st.text_input("Correo Institucional")
            st.text_input("Contraseña", type="password")
            st.button("Entrar", type="primary")
            
        with tab_reg:
            st.markdown("**Crear Cuenta Nueva**")
            st.text_input("Nombre Completo")
            st.text_input("Nuevo Correo Duoc")
            st.text_input("Crear Clave", type="password")
            st.text_input("Confirmar Clave", type="password")
            st.button("Registrar Usuario")
            st.success("☝️ ¡Captura este formulario para mostrar el registro!")

    else: # MODO ADMIN LOGUEADO
        st.info("👤 **Admin Conectado:**\ncoordinador@duoc.cl")
        st.button("Cerrar Sesión")
        st.divider()
        st.markdown("### 🛠 Herramientas Dev")
        st.checkbox("Modo Debug", value=True)
        st.checkbox("Mostrar JSON Raw")

# --- ÁREA PRINCIPAL ---
col1, col2 = st.columns([0.1, 0.9])
with col1: st.write("🤖")
with col2: st.markdown("<div class='main-header'>Plataforma de Asistencia Académica</div>", unsafe_allow_html=True)

# PESTAÑAS
tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📅 Inscripción", "🛡️ Panel Admin"])

with tab1:
    st.info("El chat está desactivado en esta vista administrativa.")

with tab2:
    st.info("El módulo de inscripción está oculto para administradores.")

# --- AQUÍ ESTÁ LO IMPORTANTE PARA TU SCREENSHOT ---
with tab3:
    if app_mode == "Vista: Registro (Logout)":
        st.error("⛔ Acceso Denegado: Debes iniciar sesión como Administrador para ver este panel.")
        st.image("https://cdn-icons-png.flaticon.com/512/675/675564.png", width=100)
    else:
        st.markdown("### 📊 Dashboard de Control y Auditoría")
        st.markdown("Monitoreo en tiempo real de interacciones y registros de usuarios.")
        
        # 1. MÉTRICAS (KPIs)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(label="Usuarios Registrados", value="1,245", delta="12 hoy")
        kpi2.metric(label="Interacciones Chat", value="8,430", delta="15%")
        kpi3.metric(label="Tasa de Error", value="2.1%", delta="-0.5%", delta_color="inverse")
        kpi4.metric(label="Feedback Positivo", value="4.8/5.0")
        
        st.divider()
        
        # 2. TABLA DE LOGS (Con estilo dataframe)
        c_table, c_details = st.columns([2, 1])
        
        with c_table:
            st.subheader("📋 Últimos Logs del Sistema")
            df_logs = get_mock_logs()
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
        
        with c_details:
            st.subheader("⚙️ Configuración Global")
            with st.container(border=True):
                st.toggle("Mantenimiento Programado", value=False)
                st.toggle("Restringir Accesos Externos", value=True)
                st.select_slider("Límite de Tokens (IA)", options=[1000, 2000, 4000, 8000], value=4000)
                st.button("🔄 Purgar Caché", help="Limpia la memoria temporal")
                
            st.warning("⚠️ **Alerta:** Se detectó un alto volumen de consultas sobre 'Exámenes Transversales' en la última hora.")

        # 3. BOTÓN DE DESCARGA
        st.download_button("📥 Descargar Reporte Completo (.csv)", data="mock_data", file_name="reporte_admin.csv")