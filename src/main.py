import os
import streamlit as st
from ml_utility import read_data, preprocess_data, train_model, evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Set Streamlit page configuration
st.set_page_config(page_title="AutoML Platform", layout="wide")
st.title("🤖 AutoML Platform")
st.markdown("Upload your dataset, select your target column, choose a model, and evaluate automatically.")

# Directory for trained models
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)
trained_model_dir = os.path.join(parent_dir, "trained_model")

# Create the trained_model folder if it doesn't exist
if not os.path.exists(trained_model_dir):
    os.makedirs(trained_model_dir)

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx", "xls", "tsv", "json"])

if uploaded_file is not None:
    try:
        # Read dataset
        df = read_data(uploaded_file)
        st.success(f"✅ Dataset uploaded successfully! Shape: {df.shape}")
        st.dataframe(df.head())

        # Select target column
        target_column = st.selectbox("🎯 Select the target column", df.columns)

        # Choose scaler
        scaler_type = st.radio("⚙️ Select feature scaling method", ["standard", "minmax"])

        # Define available models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Support Vector Machine": SVC(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier()
        }

        # Select model
        model_name = st.selectbox("🤖 Select a Machine Learning model", list(models.keys()))
        model = models[model_name]

        # Train & Evaluate Button
        if st.button("🚀 Train & Evaluate Model"):
            with st.spinner("Processing dataset and training model..."):
                X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)
                trained_model = train_model(X_train, y_train, model, model_name)
                accuracy = evaluate_model(trained_model, X_test, y_test)

            # Show training success message
            st.success(f"✅ {model_name} trained successfully! Accuracy: **{accuracy * 100:.2f}%**")

            # Path to trained model
            model_path = os.path.join(trained_model_dir, f"{model_name}.pkl")

            # Check if the model exists before showing the button
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Trained Model",
                        data=f,
                        file_name=f"{model_name}.pkl",
                        mime="application/octet-stream"
                    )
            else:
                st.warning("⚠️ Trained model file not found.")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

else:
    st.info("👆 Upload a CSV, Excel, TSV, or JSON file to get started.")
