import streamlit as st
import pandas as pd
import pickle

# Load the model
def load_model():
    with open("original_rf_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit app layout
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# Sidebar
st.sidebar.title("Sentiment Analysis Tool")
st.sidebar.write("Upload a CSV file with customer reviews and get predictions (Negative, Neutral, Positive).")

# Main title
st.title("Customer Reviews Sentiment Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with customer reviews", type=["csv"])

if uploaded_file:
    # Process the uploaded CSV file
    try:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(data.head())

        # Input for the column name
        review_column = st.text_input("Enter the column name containing the reviews:", value="review")

        if review_column not in data.columns:
            st.error(f"Column '{review_column}' not found in the uploaded file. Please check the column name.")
        else:
            # Handle missing values and predict sentiments
            reviews = data[review_column].fillna("")
            predictions = model.predict(reviews)

            # Add a new column for predictions
            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Adjust labels based on your model
            data["Sentiment"] = [sentiment_map[pred] for pred in predictions]

            # Display the resulting table
            st.write("### Sentiment Analysis Results")
            st.dataframe(data[["Sentiment"]].join(data[review_column]))

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
else:
    st.info("Please upload a CSV file to begin.")

# Footer
st.markdown(
    """
    ---
    **Developed by:** Roberto , Ignacio and Jesus 
    Powered by [Streamlit](https://streamlit.io)
    """
)
