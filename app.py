import streamlit as st
import pandas as pd
import joblib

# Configuración inicial de la página
st.set_page_config(page_title="Sentiment Analysis with TF-IDF", layout="wide")

st.title("Customer Reviews Sentiment Analysis")

# Carga del modelo y el vectorizador
model_path = st.file_uploader("Upload your Random Forest Model (.pkl)", type=["pkl"])
tfidf_path = st.file_uploader("Upload your TF-IDF Vectorizer (.pkl)", type=["pkl"])

if model_path and tfidf_path:
    try:
        rf_model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        st.success("Model and TF-IDF Vectorizer loaded successfully.")
    except Exception as e:
        st.error(f"Error loading files: {e}")
        rf_model = None
        tfidf = None
else:
    st.warning("Please upload both the Random Forest model and the TF-IDF vectorizer.")

# Carga del archivo CSV
uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=["csv"])

if uploaded_file and rf_model and tfidf:
    try:
        # Leer el archivo CSV
        data = pd.read_csv(uploaded_file)
        
        if 'reviews.text' not in data.columns:
            st.error("The CSV file must have a 'reviews.text' column for the reviews.")
        else:
            # Preprocesar y predecir
            X_new = tfidf.transform(data['reviews.text'].fillna(""))
            predictions = rf_model.predict(X_new)

            # Mapear las predicciones
            sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
            data['Predicted Sentiment'] = [sentiment_mapping[label] for label in predictions]

            # Resumir los resultados y quitar índices
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
    st.info("Please upload your model, vectorizer, and CSV file to proceed.")
