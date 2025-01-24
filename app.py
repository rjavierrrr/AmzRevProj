import streamlit as st
import pandas as pd
import gdown
import zipfile
from transformers import T5Tokenizer, T5ForConditionalGeneration
import joblib

# Configuración inicial de la página
st.set_page_config(page_title="Category and Rating Summarization", layout="wide")
st.title("Summarize Reviews by Category and Rating")

# URL del modelo de resumen en Google Drive
SUMMARIZATION_ZIP_URL = "https://drive.google.com/uc?id=1OlArb8SDWtSMdOvfMgEB1_a_6Z06I6fG"  # Reemplaza con el ID del archivo ZIP
MODEL_URL = "https://drive.google.com/uc?id=1JL0zT9kb3lwb9Rz_jwZgmg_znZWQ43oD"
TFIDF_URL = "https://drive.google.com/uc?id=1_3xaYyWWaUVaQYrXCaA2POhfIIgFx62k"

# Descargar y cargar modelos
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

@st.cache_resource
def load_sentiment_model_and_vectorizer():
    # Descargar el modelo de Random Forest
    model_output = "original_rf_model.pkl"
    gdown.download(MODEL_URL, model_output, quiet=False)
    rf_model = joblib.load(model_output)

    # Descargar el vectorizador TF-IDF
    vectorizer_output = "tfidf_vectorizer.pkl"
    gdown.download(TFIDF_URL, vectorizer_output, quiet=False)
    tfidf_vectorizer = joblib.load(vectorizer_output)

    return rf_model, tfidf_vectorizer

# Cargar los modelos
rf_model, tfidf = load_sentiment_model_and_vectorizer()
tokenizer, summarization_model = load_summarization_model()

st.success("Models loaded successfully.")

# Carga del archivo CSV
uploaded_file = st.file_uploader("Upload a CSV file with 'reviews.text', 'categories', and 'reviews.rating' columns", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        if not all(col in data.columns for col in ['reviews.text', 'categories', 'reviews.rating']):
            st.error("The CSV must have 'reviews.text', 'categories', and 'reviews.rating' columns.")
            st.stop()

        # Análisis de sentimientos
        if 'reviews.text' in data.columns:
            X_new = tfidf.transform(data['reviews.text'].fillna(""))
            predictions = rf_model.predict(X_new)

            sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
            data['Predicted Sentiment'] = [sentiment_mapping[label] for label in predictions]

            # Resumir los resultados de sentimiento
            sentiment_summary = (
                data['Predicted Sentiment']
                .value_counts()
                .reset_index()
                .rename(columns={'index': 'Sentiment', 'Predicted Sentiment': 'Count'})
            )

            st.write("### Summary of Sentiment Analysis")
            st.table(sentiment_summary)

            # Descargar los resultados
            sentiment_csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Sentiment Analysis Results as CSV",
                data=sentiment_csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )

        # Selección de cantidad de categorías y calificaciones
        unique_categories = data['categories'].unique()
        unique_ratings = [1, 2, 3, 4, 5]

        selected_categories = st.multiselect("Select Categories to Summarize", options=unique_categories, default=[])
        selected_ratings = st.multiselect("Select Ratings to Include", options=unique_ratings, default=[])

        if not selected_categories or not selected_ratings:
            st.warning("Please select at least one category and one rating.")
            st.stop()

        # Procesar y resumir por categoría y calificación
        data['reviews.text'] = data['reviews.text'].fillna("")
        summaries = []

        with st.spinner("Summarizing reviews by category and rating..."):
            for category in selected_categories:
                category_data = data[data['categories'] == category]
                for rating in selected_ratings:
                    rating_data = category_data[category_data['reviews.rating'] == rating].head(5)  # Limitar a 10 reseñas
                    if not rating_data.empty:
                        combined_text = " ".join(rating_data['reviews.text'])
                        summary = summarize_text(tokenizer, summarization_model, combined_text)
                        summaries.append({"Category": category, "Rating": rating, "Summary": summary})

        # Crear DataFrame con los resúmenes
        summary_df = pd.DataFrame(summaries)

        # Mostrar resultados
        st.write("### Summarized Reviews by Selected Categories and Ratings")
        st.dataframe(summary_df)

        # Descargar resultados como CSV
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Summarized Reviews as CSV",
            data=csv,
            file_name="summarized_reviews_by_selected_categories_and_ratings.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload your CSV file to proceed.")
