# Versión 2.1 - Reintentando LATEST con imports específicos y más depuración
import streamlit as st
import os
import langchain # Para verificar la versión

# --- Early Check and Error Reporting ---
try:
    st.write(f"Intentando cargar LangChain v{langchain.__version__}")
    from langchain_groq import ChatGroq
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers.ensemble import EnsembleRetriever # Ruta específica
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    st.write("Importaciones básicas de LangChain exitosas.")
except ImportError as e:
    st.error(f"Error CRÍTICO al importar módulos de LangChain: {e}")
    st.exception(e)
    st.stop()
except Exception as e:
    st.error(f"Error inesperado durante las importaciones: {e}")
    st.exception(e)
    st.stop()


# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Chatbot Académico Duoc UC", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot del Reglamento Académico")
st.write(f"Versión de LangChain detectada: {langchain.__version__}")

# --- CARGA DE LA API KEY DE GROQ ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("La clave de API de Groq no está configurada. Por favor, agrégala a los Secrets de Streamlit.")
    st.stop()

# --- CACHING DE RECURSOS ---
@st.cache_resource
def inicializar_cadena():
    try:
        st.write(f"Dentro de inicializar_cadena() con LangChain v{langchain.__version__}")

        # --- 1. Cargar y Procesar el PDF ---
        loader = PyPDFLoader("reglamento.pdf")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = loader.load_and_split(text_splitter=text_splitter)
        st.write(f"PDF procesado en {len(docs)} fragmentos.")

        # --- 2. Crear los Embeddings y el Ensemble Retriever ---
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(docs, embeddings)
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 7})
        doc_texts = [doc.page_content for doc in docs]
        bm25_retriever = BM25Retriever.from_texts(doc_texts)
        bm25_retriever.k = 7
        retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.7, 0.3])
        st.write("Retrievers creados exitosamente.")

        # --- 3. Conectarse al Modelo en Groq Cloud ---
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            temperature=0.1
        )
        st.write("Conexión con Groq establecida.")

        # --- 4. Crear la Cadena de Conversación ---
        prompt = ChatPromptTemplate.from_template("""
        INSTRUCCIÓN PRINCIPAL: Responde SIEMPRE en español.
        Eres un asistente experto en el reglamento académico de Duoc UC. Tu objetivo es dar respuestas claras y precisas basadas ÚNICAMENTE en el contexto proporcionado.
        Si la pregunta es general sobre "qué debe saber un alumno nuevo", crea un resumen que cubra los puntos clave: Asistencia, Calificaciones para aprobar, y Causas de Reprobación.

        CONTEXTO:
        {context}

        PREGUNTA:
        {input}

        RESPUESTA:
        """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        st.write("Cadena de LangChain creada exitosamente.")

        return retrieval_chain
    except ImportError as e:
        st.error(f"Error de importación DENTRO de inicializar_cadena: {e}")
        st.exception(e)
        return None
    except Exception as e:
        st.error(f"Error inesperado DENTRO de inicializar_cadena: {e}")
        st.exception(e)
        return None

# --- LÓGICA DE LA APLICACIÓN DE CHAT ---
retrieval_chain = inicializar_cadena()

if retrieval_chain: # Solo continúa si la inicialización fue exitosa
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("¿Qué duda tienes sobre el reglamento?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando... 💭"):
                try:
                    response = retrieval_chain.invoke({"input": prompt})
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                except Exception as e:
                    st.error(f"Error al invocar la cadena: {e}")
                    st.exception(e)
else:
    st.error("La inicialización de la cadena de LangChain falló. Revisa los errores anteriores.")