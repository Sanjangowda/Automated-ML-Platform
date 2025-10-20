import streamlit as st
from firebase_auth import signup_user, login_user
from ml_utility import read_data, preprocess_data, train_model, evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
import pandas as pd
import io
import csv

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AutoML Platform", page_icon="🤖", layout="wide")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #141e30, #243b55);
        font-family: 'Poppins', sans-serif;
        color: #fff;
    }
    .login-card:hover { transform: scale(1.02); }
    .login-card img {
        margin-bottom: 15px;
        border-radius: 50%;
        background: #ffffff20;
        padding: 10px;
    }
    .stTextInput>div>div>input {
        background-color: rgba(25, 31, 52, 0.1);
        color: white;
        border-radius: 10px;
        height: 45px;
        border: none;
        padding-left: 15px;
        font-size: 15px;
    }
    .stTextInput>div>div>input::placeholder {
        color: rgba(255, 255, 255, 0.7);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 12px;
        padding: 10px 30px;
        font-weight: 600;
        letter-spacing: 1px;
        margin-top: 20px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        background: linear-gradient(90deg, #0072ff, #00c6ff);
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #253342;
        font-weight: 500;
        padding: 10px 25px;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: #fff;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Authentication
# -----------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "datasets" not in st.session_state:
    st.session_state.datasets = []
if "models" not in st.session_state:
    st.session_state.models = []

if not st.session_state.user:
    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Sign Up"])

    # ----------------- Login -----------------
    with tab1:
        st.image("https://cdn-icons-png.flaticon.com/512/847/847969.png", width=80)
        st.markdown("<h2>Welcome Back 👋</h2>", unsafe_allow_html=True)
        email = st.text_input("📧 Email ID", key="login_email", placeholder="Enter your email")
        password = st.text_input("🔒 Password", type="password", key="login_password", placeholder="Enter your password")

        if st.button("LOGIN"):
            user = login_user(email, password)
            if isinstance(user, dict) and "idToken" in user:
                st.session_state.user = {"email": user["email"], "idToken": user["idToken"]}
                st.success("✅ Logged in successfully!")
                st.rerun()
            else:
                st.error(f"⚠️ {user}")

    # ----------------- Signup -----------------
    with tab2:
        st.image("https://cdn-icons-png.flaticon.com/512/1828/1828469.png", width=80)
        st.markdown("<h2>Create Account ✨</h2>", unsafe_allow_html=True)
        email = st.text_input("📧 Email ID", key="signup_email", placeholder="Enter your email")
        password = st.text_input("🔒 Password", type="password", key="signup_password", placeholder="Create a password")

        if st.button("SIGN UP"):
            user = signup_user(email, password)
            if isinstance(user, dict) and "idToken" in user:
                st.success("✅ Account created successfully! Please login.")
            else:
                st.error(f"⚠️ {user}")

else:
    # -----------------------------
    # Sidebar
    # -----------------------------
    st.sidebar.success(f"👤 Logged in as {st.session_state.user['email']}")
    uploaded_file = st.sidebar.file_uploader("📂 Upload your dataset", type=["csv", "xlsx", "xls", "tsv", "json"])
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    # -----------------------------
    # Tabs
    # -----------------------------
    st.title("🤖 AutoML Platform")
    tab_profile, tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Profile Dashboard",
        "📊 EDA",
        "⚙️ Train Model",
        "📥 Predictions",
        "💾 Download Models"
    ])

    # ---------------- Profile Dashboard ----------------
    with tab_profile:
        st.header("👤 Profile Dashboard")
        st.write(f"**Email:** {st.session_state.user['email']}")
        st.markdown("### 📂 Uploaded Datasets")
        if st.session_state.datasets:
            st.table(pd.DataFrame(st.session_state.datasets, columns=["Dataset Name"]))
        else:
            st.info("No datasets uploaded yet.")
        st.markdown("### 🤖 Trained Models")
        if st.session_state.models:
            st.table(pd.DataFrame(st.session_state.models))
        else:
            st.info("No models trained yet.")

    # ---------------- Dataset Tabs ----------------
    if uploaded_file:
        try:
            df = read_data(uploaded_file)
            if uploaded_file.name not in st.session_state.datasets:
                st.session_state.datasets.append(uploaded_file.name)

            # --------------- EDA ----------------
            with tab1:
                st.subheader("📊 Exploratory Data Analysis")
                st.write("**Dataset Shape:**", df.shape)
                st.write("**Columns:**", df.columns.tolist())
                st.dataframe(df.head())
                st.write("**Summary Statistics:**")
                st.write(df.describe(include="all"))

            # --------------- Train ----------------
            with tab2:
                st.subheader("⚙️ Train a Machine Learning Model")
                target_column = st.selectbox("🎯 Select target column", df.columns)
                scaler_type = st.radio("⚖️ Scaling", ["standard", "minmax"], horizontal=True)

                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Support Vector Machine": SVC(),
                    "Random Forest": RandomForestClassifier(),
                    "XGBoost": XGBClassifier()
                }

                model_name = st.selectbox("🤖 Choose Model", list(models.keys()))
                trained_model_name = st.text_input("💾 Name for trained model", "my_model")

                if st.button("🚀 Train & Evaluate"):
                    with st.spinner("🔄 Processing..."):
                        X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)
                        trained_model = train_model(X_train, y_train, models[model_name], trained_model_name)
                        accuracy = evaluate_model(trained_model, X_test, y_test)

                    st.success(f"✅ {model_name} trained! Accuracy: **{accuracy * 100:.2f}%**")

                    model_file = f"{trained_model_name}.pkl"
                    with open(model_file, "wb") as f:
                        pickle.dump(trained_model, f)
                    st.session_state["last_model_file"] = model_file

                    st.session_state.models.append({
                        "name": trained_model_name,
                        "type": model_name,
                        "accuracy": f"{accuracy*100:.2f}%",
                        "file": model_file
                    })

            # --------------- Predictions ----------------
            with tab3:
                st.subheader("📥 Predictions using Saved Test Dataset")

                import os

                test_data_path = os.path.join("test_dataset", "test_data.csv")

                if not os.path.exists(test_data_path):
                    st.warning("⚠️ Please train a model first to generate a test dataset.")
                elif not st.session_state.models:
                    st.warning("⚠️ No trained models found. Train one to continue.")
                else:
                    # Select trained model
                    model_choice = st.selectbox(
                        "🤖 Select a trained model",
                        [m["name"] for m in st.session_state.models]
                    )
                    selected_model_info = next((m for m in st.session_state.models if m["name"] == model_choice), None)
                    model_file = selected_model_info["file"]

                    # Load test dataset
                    try:
                        df_test = pd.read_csv(test_data_path)
                        st.success("✅ Test dataset loaded successfully!")
                        st.write("### Test Dataset Preview")
                        st.dataframe(df_test.head())

                        # Get target column (last column)
                        target_col = df_test.columns[-1]
                        X_test = df_test.drop(columns=[target_col])
                        y_test = df_test[target_col]

                        if st.button("🔮 Run Predictions"):
                            with st.spinner("Generating predictions..."):
                                # Load trained model
                                with open(model_file, "rb") as f:
                                    model = pickle.load(f)

                                # Predict
                                preds = model.predict(X_test)

                                # Create result DataFrame
                                result_df = pd.DataFrame({
                                    "Actual": y_test,
                                    "Predicted": preds
                                })
                                result_df["Status"] = result_df.apply(
                                    lambda row: "✅ Correct" if row["Actual"] == row["Predicted"] else "❌ Incorrect",
                                    axis=1
                                )

                                # Apply color highlighting
                                def highlight_rows(row):
                                    color = "background-color: #d1f7c4;" if row["Status"] == "✅ Correct" else "background-color: #f8d7da;"
                                    return [color] * len(row)

                                styled_df = result_df.style.apply(highlight_rows, axis=1)

                                st.success(f"✅ Predictions completed using **{model_choice}**!")
                                st.write("### 🎯 Prediction Results")
                                st.dataframe(styled_df, use_container_width=True)

                                # Evaluation metrics
                                try:
                                    from sklearn.metrics import accuracy_score, mean_absolute_error
                                    if y_test.dtypes == "int" or y_test.dtypes == "float":
                                        acc = accuracy_score(y_test, preds)
                                        mae = mean_absolute_error(y_test, preds)
                                        st.markdown(f"**📊 Accuracy:** `{acc * 100:.2f}%`")
                                        st.markdown(f"**📉 Mean Absolute Error:** `{mae:.4f}`")
                                except Exception:
                                    pass

                                # Download results
                                csv = result_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Predictions",
                                    data=csv,
                                    file_name="predictions.csv",
                                    mime="text/csv"
                                )

                    except Exception as e:
                        st.error(f"⚠️ Error loading or predicting: {str(e)}")

            # --------------- Download ----------------
            with tab4:
                st.subheader("💾 Download Trained Models")
                if "last_model_file" in st.session_state:
                    st.download_button(
                        "📥 Download Latest Model",
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
