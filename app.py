import streamlit as st
import pandas as pd
import joblib
import gdown
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Configuración inicial de la página
st.set_page_config(page_title="Sentiment Analysis & Summarization", layout="wide")
st.title("Customer Reviews Sentiment Analysis and Summarization")

# URLs del modelo
MODEL_URL = "https://drive.google.com/uc?id=1JL0zT9kb3lwb9Rz_jwZgmg_znZWQ43oD"
TFIDF_URL = "https://drive.google.com/uc?id=1_3xaYyWWaUVaQYrXCaA2POhfIIgFx62k"
SUMMARIZATION_MODEL_DIR = "./flan_t5_summary_model"

# Cargar el modelo de resumen
@st.cache_resource
def load_summarization_model():
    tokenizer = T5Tokenizer.from_pretrained(SUMMARIZATION_MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(SUMMARIZATION_MODEL_DIR)
    return tokenizer, model

# Función para resumir reseñas
def summarize_review(tokenizer, model, review_text):
    inputs = tokenizer.encode("summarize: " + review_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Descargar y cargar modelo y vectorizador
@st.cache_resource
def load_model_and_vectorizer():
    model_output = "original_rf_model.pkl"
    gdown.download(MODEL_URL, model_output, quiet=False)
    rf_model = joblib.load(model_output)

    vectorizer_output = "tfidf_vectorizer.pkl"
    gdown.download(TFIDF_URL, vectorizer_output, quiet=False)
    tfidf_vectorizer = joblib.load(vectorizer_output)

    return rf_model, tfidf_vectorizer

rf_model, tfidf = load_model_and_vectorizer()
tokenizer, summarization_model = load_summarization_model()

st.success("Models loaded successfully.")

# Carga del archivo CSV
uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        if 'reviews.text' not in data.columns or 'reviews.rating' not in data.columns or 'product.category' not in data.columns:
            st.error("The CSV must have 'reviews.text', 'reviews.rating', and 'product.category' columns.")
            st.stop()

        # Procesar y predecir sentimientos
        data['reviews.text'] = data['reviews.text'].fillna("")
        X_new = tfidf.transform(data['reviews.text'])
        predictions = rf_model.predict(X_new)

        sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        data['Predicted Sentiment'] = [sentiment_mapping[label] for label in predictions]

        # Resumen por categorías y calificaciones
        st.write("### Summarized Reviews by Category and Rating")
        top_k_categories = st.slider("Select number of categories to summarize", min_value=1, max_value=50, value=10)
        category_summary = []

        top_categories = data['product.category'].value_counts().head(top_k_categories).index
        for category in top_categories:
            st.write(f"#### Product Category: {category}")
            category_data = data[data['product.category'] == category]

            for rating in range(1, 6):  # Rating 1 to 5
                reviews = category_data[category_data['reviews.rating'] == rating]['reviews.text']
                if not reviews.empty:
                    combined_text = " ".join(reviews)
                    summary = summarize_review(tokenizer, summarization_model, combined_text)
                    st.write(f"**Rating {rating} Summary:** {summary}")
                    category_summary.append({"Category": category, "Rating": rating, "Summary": summary})

        # Descargar resultados
        summary_df = pd.DataFrame(category_summary)
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Summarization Results as CSV",
            data=csv,
            file_name="summarized_reviews.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload your CSV file to proceed.")