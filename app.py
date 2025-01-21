import streamlit as st
import pandas as pd
import pickle
import os

# Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# Function to load the model
def load_model():
    model_path = "original_rf_model.pkl"

    # Check if the file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")

    # Check if the file is not empty
    if os.path.getsize(model_path) == 0:
        raise ValueError(f"The model file {model_path} is empty.")

    # Attempt to load the model
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Attempt to load the model
try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

# Streamlit app layout
st.sidebar.title("Sentiment Analysis Tool")
st.sidebar.write("Upload a CSV file with customer reviews to classify them as Negative, Neutral, or Positive.")

st.title("Customer Reviews Sentiment Analysis")

# File uploader for the CSV file
uploaded_file = st.file_uploader("Upload a CSV file with customer reviews", type=["csv"])

if uploaded_file and model:
    try:
        # Load and display the CSV file
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(data.head())

        # Specify the review column
        review_column = st.text_input("Enter the column name containing the reviews:", value="review")

        if review_column not in data.columns:
            st.error(f"Column '{review_column}' not found in the uploaded file. Please check the column name.")
        else:
            # Process and predict sentiments
            reviews = data[review_column].fillna("")
            predictions = model.predict(reviews)

            # Map predictions to labels
            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Adjust based on your model's output
            data["Sentiment"] = [sentiment_map[pred] for pred in predictions]

            st.write("### Sentiment Analysis Results")
            st.dataframe(data[[review_column, "Sentiment"]])

            # Downloadable CSV
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
