import streamlit as st
import pandas as pd
import pickle

# Streamlit app layout
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# Sidebar
st.sidebar.title("Sentiment Analysis Tool")
st.sidebar.write("Upload your model and a CSV file with customer reviews to classify them as Negative, Neutral, or Positive.")

# Main title
st.title("Customer Reviews Sentiment Analysis")

# File uploader for the model
st.write("### Step 1: Upload your Model File (.pkl)")
uploaded_model = st.file_uploader("Upload your model file", type=["pkl"], key="model_file")

# Placeholder for the model
model = None

if uploaded_model:
    try:
        # Load the uploaded model
        model = pickle.load(uploaded_model)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading the model: {e}")

# File uploader for the CSV file
st.write("### Step 2: Upload your CSV File with Reviews")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="csv_file")

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

            # Display results
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
