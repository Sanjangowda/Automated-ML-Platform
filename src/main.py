import streamlit as st
from firebase_auth import signup_user, login_user
from ml_utility import read_data, preprocess_data, train_model, evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
import pandas as pd


# Page Config

st.set_page_config(page_title="AutoML Platform", page_icon="🤖", layout="wide")


# Custom CSS

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        h1, h2, h3 {
            font-family: 'Segoe UI', sans-serif;
            font-weight: 600;
        }
        .stButton>button {
            border-radius: 10px;
            background-color: #4CAF50;
            color: white;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stDownloadButton>button {
            border-radius: 10px;
            background-color: #1E90FF;
            color: white;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
        }
        .stDownloadButton>button:hover {
            background-color: #1C86EE;
        }
        .stSelectbox label, .stRadio label {
            font-weight: 600 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Authentication

if "user" not in st.session_state:
    st.session_state.user = None
if "datasets" not in st.session_state:
    st.session_state.datasets = []
if "models" not in st.session_state:
    st.session_state.models = []

if not st.session_state.user:
    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Sign Up"])

    with tab1:
        st.subheader("Login to your account")
        email = st.text_input(" Email", key="login_email")
        password = st.text_input("🔒 Password", type="password", key="login_password")
        if st.button("Login"):
            user = login_user(email, password)
            if isinstance(user, dict) and "idToken" in user:
                st.session_state.user = {"email": user["email"], "idToken": user["idToken"]}
                st.success("✅ Logged in successfully!")
                st.rerun()
            else:
                st.error(f"⚠️ {user}")

    with tab2:
        st.subheader("Create a new account")
        email = st.text_input(" Email", key="signup_email")
        password = st.text_input("🔒 Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            user = signup_user(email, password)
            if isinstance(user, dict) and "idToken" in user:
                st.success("✅ Account created successfully! Please login.")
            else:
                st.error(f"⚠️ {user}")

else:
 
    # Sidebar (only essentials)
   
    st.sidebar.success(f"👤 Logged in as {st.session_state.user['email']}")
    uploaded_file = st.sidebar.file_uploader("📁 Upload your dataset", type=["csv", "xlsx", "xls", "tsv", "json"])
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

  
    # Dashboard Tabs

    st.title("🤖 AutoML Platform")
    tab_profile, tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Profile Dashboard",
        "📊 EDA",
        "⚙️ Train Model",
        "💡 Predictions",
        "📥 Download Models"
    ])

    # Profile Dashboard 
    with tab_profile:
        st.header("👤 Profile Dashboard")
        st.write(f"**Email:** {st.session_state.user['email']}")

        st.markdown("### 📑 Uploaded Datasets")
        if st.session_state.datasets:
            st.table(pd.DataFrame(st.session_state.datasets, columns=["Dataset Name"]))
        else:
            st.info("No datasets uploaded yet.")

        st.markdown("### 🤖 Trained Models")
        if st.session_state.models:
            st.table(pd.DataFrame(st.session_state.models))
        else:
            st.info("No models trained yet.")

    # Dataset Required Tabs 
    if uploaded_file:
        try:
            df = read_data(uploaded_file)

            # Track dataset
            if uploaded_file.name not in st.session_state.datasets:
                st.session_state.datasets.append(uploaded_file.name)

            # EDA 
            with tab1:
                st.subheader("📊 Exploratory Data Analysis")
                st.write("**Dataset Shape:**", df.shape)
                st.write("**Columns:**", df.columns.tolist())
                st.dataframe(df.head())
                st.write("**Summary Statistics:**")
                st.write(df.describe(include="all"))

            # Training Model 
            with tab2:
                st.subheader("⚙️ Train a Machine Learning Model")

                target_column = st.selectbox("🎯 Select the target column", df.columns)
                scaler_type = st.radio("⚖️ Feature scaling method", ["standard", "minmax"], horizontal=True)

                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Support Vector Machine": SVC(),
                    "Random Forest": RandomForestClassifier(),
                    "XGBoost": XGBClassifier()
                }
                model_name = st.selectbox("🤖 Choose Model", list(models.keys()))
                trained_model_name = st.text_input("📝 Name for trained model", "my_model")

                if st.button("🚀 Train & Evaluate"):
                    with st.spinner("🔄 Processing dataset..."):
                        X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)
                        trained_model = train_model(X_train, y_train, models[model_name], trained_model_name)
                        accuracy = evaluate_model(trained_model, X_test, y_test)

                    st.success(f"✅ {model_name} trained successfully! Accuracy: **{accuracy * 100:.2f}%**")

                    # Save model
                    model_file = f"{trained_model_name}.pkl"
                    with open(model_file, "wb") as f:
                        pickle.dump(trained_model, f)
                    st.session_state["last_model_file"] = model_file

                    # Track trained model
                    st.session_state.models.append({
                        "name": trained_model_name,
                        "type": model_name,
                        "accuracy": f"{accuracy*100:.2f}%",
                        "file": model_file
                    })

            # Predictions
            with tab3:
                st.subheader("Make Predictions (Coming Soon 🚧)")
                st.info("Future enhancement: Allow users to upload new data and get predictions.")

            #  Download Models
            with tab4:
                st.subheader("📥 Download Trained Models")
                if "last_model_file" in st.session_state:
                    st.download_button(
                        "⬇️ Download Latest Model",
                        data=open(st.session_state["last_model_file"], "rb").read(),
                        file_name=st.session_state["last_model_file"],
                        mime="application/octet-stream"
                    )
                else:
                    st.info("No model trained yet.")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
    else:
        st.info("👈 Upload a dataset from the sidebar to begin.")
