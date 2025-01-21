import streamlit as st
import pandas as pd
import pickle

# Configuración inicial de la página
st.set_page_config(page_title="Customer Reviews Sentiment Analysis", layout="wide")

# Título de la aplicación
st.title("Customer Reviews Sentiment Analysis")

# Paso 1: Cargar el modelo manualmente
st.write("### Step 1: Upload your Model File (.pkl)")
uploaded_model = st.file_uploader("Upload your model file", type=["pkl"], key="model_file")

model = None
if uploaded_model:
    try:
        # Intentar cargar el modelo
        model = pickle.load(uploaded_model)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
else:
    st.warning("Please upload your model file to proceed.")

# Paso 2: Cargar un archivo CSV
st.write("### Step 2: Upload your CSV File with Reviews")
uploaded_file = st.file_uploader("Upload a CSV file with customer reviews", type=["csv"], key="csv_file")

if uploaded_file and model:
    try:
        # Leer el archivo CSV
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(data.head())

        # Especificar la columna que contiene las reseñas
        review_column = st.text_input("Enter the column name containing the reviews:", value="review")

        if review_column not in data.columns:
            st.error(f"Column '{review_column}' not found in the uploaded file. Please check the column name.")
        else:
            # Predecir sentimientos
            reviews = data[review_column].fillna("")
            predictions = model.predict(reviews)

            # Mapear los resultados a etiquetas de sentimiento
            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Ajusta según la salida de tu modelo
            data["Sentiment"] = [sentiment_map[pred] for pred in predictions]

            # Mostrar los resultados
            st.write("### Sentiment Analysis Results")
            st.dataframe(data[[review_column, "Sentiment"]])

            # Botón para descargar los resultados
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing the file: {e}")
elif not model:
    st.warning("Please upload your model file to proceed.")
