import streamlit as st
import pandas as pd
import joblib
import gdown
from transformers import pipeline

# Configuración inicial de la página
st.set_page_config(page_title="Sentiment Analysis with Summarization", layout="wide")

st.title("Customer Reviews Sentiment Analysis and Summarization")

# URLs de los archivos en Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1JL0zT9kb3lwb9Rz_jwZgmg_znZWQ43oD"
TFIDF_URL = "https://drive.google.com/uc?id=1_3xaYyWWaUVaQYrXCaA2POhfIIgFx62k"

# Función para descargar y cargar el modelo de análisis de sentimientos
@st.cache_resource
def load_model_and_vectorizer():
    # Descargar el modelo de Random Forest
    model_output = "original_rf_model.pkl"
    gdown.download(MODEL_URL, model_output, quiet=False)
    rf_model = joblib.load(model_output)

    # Descargar el vectorizador TF-IDF
    vectorizer_output = "tfidf_vectorizer.pkl"
    gdown.download(TFIDF_URL, vectorizer_output, quiet=False)
    tfidf_vectorizer = joblib.load(vectorizer_output)

    return rf_model, tfidf_vectorizer

# Función para cargar el modelo de resumen
@st.cache_resource
def load_summarization_model():
    summarizer = pipeline("summarization", model="google/flan-t5-base")
    return summarizer

# Cargar modelos
try:
    rf_model, tfidf = load_model_and_vectorizer()
    summarizer = load_summarization_model()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Carga del archivo CSV
uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=["csv"])

if uploaded_file:
    try:
        # Leer el archivo CSV
        data = pd.read_csv(uploaded_file)

        if 'reviews.text' not in data.columns:
            st.error("The CSV file must have a 'reviews.text' column for the reviews.")
        else:
            # Preprocesar las reseñas y realizar predicciones
            X_new = tfidf.transform(data['reviews.text'].fillna(""))
            predictions = rf_model.predict(X_new)

            # Mapear las predicciones a etiquetas
            sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
            data['Predicted Sentiment'] = [sentiment_mapping[label] for label in predictions]

            # Resumir los resultados
            sentiment_summary = (
                data['Predicted Sentiment']
                .value_counts()
                .reset_index()
                .rename(columns={'index': 'Sentiment', 'Predicted Sentiment': 'Count'})
            )

            # Mostrar la tabla resumen
            st.write("### Summary of Sentiment Analysis")
            st.table(sentiment_summary)

            # Agrupar las reseñas por sentimiento y resumirlas
            st.write("### Summarizing Reviews by Sentiment")
            grouped_reviews = (
                data.groupby('Predicted Sentiment')['reviews.text']
                .apply(lambda x: " ".join(x))
                .reset_index()
                .rename(columns={'reviews.text': 'All Reviews'})
            )

            grouped_reviews['Summary'] = grouped_reviews['All Reviews'].apply(
                lambda x: summarizer(x, max_length=100, min_length=30, truncation=True)[0]['summary_text']
            )

            # Mostrar los resúmenes
            st.write("### Review Summaries")
            st.dataframe(grouped_reviews[['Predicted Sentiment', 'Summary']])

            # Descargar los resultados
            output_file = "summarized_reviews.csv"
            grouped_reviews.to_csv(output_file, index=False)
            st.download_button(
                label="Download Summarized Reviews as CSV",
                data=open(output_file, "rb").read(),
                file_name=output_file,
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
