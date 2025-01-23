import streamlit as st
import pandas as pd
import joblib
import gdown
from transformers import pipeline
import plotly.express as px

# Configuración inicial de la página
st.set_page_config(page_title="Sentiment Analysis and Summarization Dashboard", layout="wide")

st.title("Customer Reviews Sentiment Analysis and Summarization Dashboard")

# URLs de los modelos en Google Drive
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

        # Verificar columnas necesarias
        required_columns = {'categories', 'reviews.rating', 'reviews.text'}
        if not required_columns.issubset(data.columns):
            st.error(f"The CSV file must contain the following columns: {required_columns}")
        else:
            # Preprocesar los datos
            data = data.dropna(subset=['categories', 'reviews.rating', 'reviews.text'])
            data = data[data['reviews.text'].str.strip() != ""]
            data['reviews.rating'] = data['reviews.rating'].astype(int)

            # Seleccionar Top-K categorías
            K = st.slider("Select number of top categories to analyze", min_value=1, max_value=20, value=10)
            top_categories = data['categories'].value_counts().nlargest(K).index
            filtered_data = data[data['categories'].isin(top_categories)]

            # Preprocesar reseñas y predecir sentimientos
            X_new = tfidf.transform(filtered_data['reviews.text'].fillna(""))
            predictions = rf_model.predict(X_new)

            # Mapear predicciones a etiquetas
            sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
            filtered_data['Predicted Sentiment'] = [sentiment_mapping[label] for label in predictions]

            # Agrupar por categoría y calificación
            grouped_reviews = (
                filtered_data
                .groupby(['categories', 'reviews.rating'])['reviews.text']
                .apply(lambda x: " ".join(x))
                .reset_index()
                .rename(columns={'reviews.text': 'All Reviews'})
            )

            # Generar resúmenes
            st.write("### Generating summaries...")
            grouped_reviews['Summary'] = grouped_reviews['All Reviews'].apply(
                lambda text: summarizer(text, max_length=100, min_length=30, truncation=True)[0]['summary_text']
            )
            st.success("Summaries generated successfully!")

            # Visualización interactiva con Plotly
            st.write("### Visualization Dashboard")
            fig = px.sunburst(
                grouped_reviews,
                path=['categories', 'reviews.rating'],
                values=None,
                color='reviews.rating',
                hover_data=['Summary'],
                title="Summary by Categories and Ratings"
            )
            st.plotly_chart(fig, use_container_width=True)

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
    st.info("Please upload a CSV file to proceed.")
