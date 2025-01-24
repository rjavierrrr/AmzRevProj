import streamlit as st
import pandas as pd
import gdown
import zipfile
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Configuración inicial de la página
st.set_page_config(page_title="Category and Rating Summarization", layout="wide")
st.title("Summarize Reviews by Category and Rating")

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

# Carga del archivo CSV
uploaded_file = st.file_uploader("Upload a CSV file with 'reviews.text', 'categories', and 'reviews.rating' columns", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        if not all(col in data.columns for col in ['reviews.text', 'categories', 'reviews.rating']):
            st.error("The CSV must have 'reviews.text', 'categories', and 'reviews.rating' columns.")
            st.stop()

        # Procesar y resumir por categoría y calificación
        data['reviews.text'] = data['reviews.text'].fillna("")
        summaries = []

        with st.spinner("Summarizing reviews by category and rating..."):
            top_categories = data['categories'].value_counts().head(5).index
            for category in top_categories:
                category_data = data[data['categories'] == category]
                for rating in range(1, 6):
                    rating_data = category_data[category_data['reviews.rating'] == rating].head(10)  # Limitar a 10 reseñas
                    if not rating_data.empty:
                        combined_text = " ".join(rating_data['reviews.text'])
                        summary = summarize_text(tokenizer, summarization_model, combined_text)
                        summaries.append({"Category": category, "Rating": rating, "Summary": summary})

        # Crear DataFrame con los resúmenes
        summary_df = pd.DataFrame(summaries)

        # Mostrar resultados
        st.write("### Summarized Reviews by Top 5 Categories and Rating")
        st.dataframe(summary_df)

        # Descargar resultados como CSV
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Summarized Reviews as CSV",
            data=csv,
            file_name="summarized_reviews_by_top_5_categories_and_rating.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload your CSV file to proceed.")
