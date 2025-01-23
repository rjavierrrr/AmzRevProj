import streamlit as st
import gdown
import zipfile
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Configuración inicial de la página
st.set_page_config(page_title="Text Summarization", layout="wide")
st.title("Text Summarization with T5 Model")

# URL del modelo de resumen en Google Drive
SUMMARIZATION_ZIP_URL = "https://drive.google.com/uc?id=1OlArb8SDWtSMdOvfMgEB1_a_6Z06I6fG"  # Reemplaza con el ID del archivo ZIP

# Descargar y cargar modelo de resumen
@st.cache_resource
def load_summarization_model():
    # Descargar y descomprimir el modelo de resumen
    summarization_zip = "flan_t5_summary_model.zip"
    gdown.download(SUMMARIZATION_ZIP_URL, summarization_zip, quiet=False)
    with zipfile.ZipFile(summarization_zip, "r") as zip_ref:
        zip_ref.extractall("./")

    tokenizer = T5Tokenizer.from_pretrained("./flan_t5_summary_model/flan_t5_summary_model")
    model = T5ForConditionalGeneration.from_pretrained("./flan_t5_summary_model/flan_t5_summary_model")
    return tokenizer, model

# Función para resumir texto
def summarize_text(tokenizer, model, input_text, max_length=150, min_length=40):
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Cargar el modelo de resumen
tokenizer, summarization_model = load_summarization_model()

st.success("Model loaded successfully.")

# Interfaz para ingresar texto y resumirlo
st.write("### Enter the text you want to summarize")
input_text = st.text_area("Input Text", height=200)

if st.button("Summarize"):
    if input_text.strip():
        with st.spinner("Summarizing..."):
            summary = summarize_text(tokenizer, summarization_model, input_text)
        st.write("### Summary")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize.")
