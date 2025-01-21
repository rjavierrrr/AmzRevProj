import streamlit as st
import pandas as pd
import joblib
import gdown

# Configuración inicial de la página
st.set_page_config(page_title="Sentiment Analysis with TF-IDF", layout="wide")

st.title("Customer Reviews Sentiment Analysis")

# URL del archivo en Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1JL0zT9kb3lwb9Rz_jwZgmg_znZWQ43oD"
TFIDF_URL = "https://drive.google.com/uc?id=1_3xaYyWWaUVaQYrXCaA2POhfIIgFx62k"  # Reemplaza con el enlace de tu TF-IDF

# Función para descargar y cargar el modelo
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

# Cargar el modelo y vectorizador
try:
    rf_model, tfidf = load_model_and_vectorizer()
    st.success("Model and TF-IDF Vectorizer loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
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

            # Descargar los resultados
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predicted_reviews.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload your CSV file to proceed.")
