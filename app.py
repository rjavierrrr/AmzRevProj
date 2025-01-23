import streamlit as st
import pandas as pd
import joblib
import gdown
from transformers import pipeline

# Configuración inicial de la página
st.set_page_config(page_title="Sentiment Analysis and Summarization", layout="wide")

st.title("Customer Reviews Sentiment Analysis and Summarization")

# URLs del modelo y el vectorizador en Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1JL0zT9kb3lwb9Rz_jwZgmg_znZWQ43oD"
TFIDF_URL = "https://drive.google.com/uc?id=1_3xaYyWWaUVaQYrXCaA2POhfIIgFx62k"

# Función para descargar y cargar el modelo de análisis de sentimientos
@st.cache_resource
def load_model_and_vectorizer():
    model_output = "original_rf_model.pkl"
    gdown.download(MODEL_URL, model_output, quiet=False)
    rf_model = joblib.load(model_output)

    vectorizer_output = "tfidf_vectorizer.pkl"
    gdown.download(TFIDF_URL, vectorizer_output, quiet=False)
    tfidf_vectorizer = joblib.load(vectorizer_output)

    return rf_model, tfidf_vectorizer

# Función para cargar el modelo de resumen desde Hugging Face
@st.cache_resource
def load_summarization_model():
    summarizer = pipeline("summarization", model="google/flan-t5-base")
    return summarizer

# Cargar los modelos
try:
    rf_model, tfidf = load_model_and_vectorizer()
    summarizer = load_summarization_model()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Cargar el archivo CSV
uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=["csv"])

if uploaded_file:
    try:
        # Leer el archivo CSV
        data = pd.read_csv(uploaded_file)

        if not {'categories', 'reviews.rating', 'reviews.text'}.issubset(data.columns):
            st.error("The dataset must contain 'categories', 'reviews.rating', and 'reviews.text' columns.")
        else:
            # Preprocesar datos
            data = data.dropna(subset=['categories', 'reviews.rating', 'reviews.text'])
            data = data[data['reviews.text'].str.strip() != ""]
            data['reviews.rating'] = data['reviews.rating'].astype(int)

            # Realizar análisis de sentimientos
            X_new = tfidf.transform(data['reviews.text'].fillna(""))
            predictions = rf_model.predict(X_new)

            sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
            data['Predicted Sentiment'] = [sentiment_mapping[label] for label in predictions]

            # Resumir los resultados
            sentiment_summary = (
                data['Predicted Sentiment']
                .value_counts()
                .reset_index()
                .rename(columns={'index': 'Sentiment', 'Predicted Sentiment': 'Count'})
            )
            st.write("### Summary of Sentiment Analysis")
            st.table(sentiment_summary)

            # Seleccionar las 10 principales categorías
            top_categories = data['categories'].value_counts().nlargest(3).index
            filtered_data = data[data['categories'].isin(top_categories)]

            # Agrupar reseñas por categoría y calificación
            grouped_reviews = (
                filtered_data.groupby(['categories', 'reviews.rating'])['reviews.text']
                .apply(lambda x: " ".join(x))
                .reset_index()
            )

            # Generar resúmenes
            st.write("### Generating Summaries for Each Category and Rating")

            def summarize_reviews(text, max_length=512):
                try:
                    return summarizer(text, max_length=max_length, min_length=30, truncation=True)[0]['summary_text']
                except Exception as e:
                    return f"Error during summarization: {str(e)}"

            grouped_reviews['Summary'] = grouped_reviews['reviews.text'].apply(summarize_reviews)
            st.success("Summaries generated successfully!")

            # Mostrar tabla de resúmenes
            st.write("### Summarized Reviews")
            st.dataframe(grouped_reviews[['categories', 'reviews.rating', 'Summary']])

            # Descargar resultados
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
    st.info("Please upload your CSV file to proceed.")
