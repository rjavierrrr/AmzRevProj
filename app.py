import streamlit as st
import pandas as pd
import pickle
import os

# Load the model
def load_model():
    model_path = "original_rf_model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")
    
    if os.path.getsize(model_path) == 0:
        raise ValueError(f"The model file {model_path} is empty.")

    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

# Streamlit app layout
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# Sidebar
st.sidebar.title("Sentiment Analysis Tool")
st.sidebar.write("Upload a CSV file with customer reviews to classify them as Negative, Neutral, or Positive.")

# Main title
st.title("Customer Reviews Sentiment Analysis")

# File uploader for the CSV file
uploaded_file = st.file_uploader("Upload a CSV file with customer reviews", type=["csv"])

if uploaded_file and model:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(data.head())

        review_column = st.text_input("Enter the column name containing the reviews:", value="review")

        if review_column not in data.columns:
            st.error(f"Column '{review_column}' not found in the uploaded file. Please check the column name.")
        else:
            reviews = data[review_column].fillna("")
            predictions = model.predict(reviews)

            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Adjust labels based on your model
            data["Sentiment"] = [sentiment_map[pred] for pred in predictions]

            st.write("### Sentiment Analysis Results")
            st.dataframe(data[[review_column, "Sentiment"]])

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
