import os
import streamlit as st
from ml_utility import read_data, preprocess_data, train_model, evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
   
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
st.markdown(
    """
    <style>
    /* Main container padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Title Styling */
    .main-title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        color: transparent;
        padding-bottom: 10px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #222;
        padding: 10px 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #333;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(to right, #0072ff, #00c6ff) !important;
        color: white !important;
    }

    /* Cards */
    .stCard {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(to right, #0072ff, #00c6ff);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.05);
        background: linear-gradient(to right, #00c6ff, #0072ff);
    }

    /* Download Button */
    div.stDownloadButton > button {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 16px;
        border: none;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #101010;
        color: white;
        padding: 20px;
    }

    /* Dataframe Scroll */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }

        /* Push the title down to avoid header overlap */
        .main-title {
            padding-top: 40px;
            font-size: 36px;
            font-weight: bold;
            color: #00aaff;
            text-align: center;
        }

        /* Adjust header spacing */
        header.css-18ni7ap.e8zbici2 {
            height: 60px;
        }

        /* Improve button styling */
        div.stButton > button {
            width: 200px;
            height: 45px;
            border-radius: 12px;
            background-color: #0078D7;
            color: white;
            font-size: 16px;
            font-weight: bold;
            transition: 0.3s;
        }

        div.stButton > button:hover {
            background-color: #005a9e;
            transform: scale(1.05);
        }

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- HEADER --------------------
st.markdown('<div class="main-title">🤖 AutoML Platform</div>', unsafe_allow_html=True)
st.markdown("### A modern AutoML dashboard to upload datasets, train models, evaluate performance, and download trained models.")

# -------------------- TRAINED MODELS DIRECTORY --------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)
trained_model_dir = os.path.join(parent_dir, "trained_model")

if not os.path.exists(trained_model_dir):
    os.makedirs(trained_model_dir)

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs(["📂 Dataset Preview", "⚙️ Model Training", "📊 Results", "📥 Download Models"])

# -------------------- TAB 1: DATASET PREVIEW --------------------
with tab1:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx", "xls", "tsv", "json"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            df = read_data(uploaded_file)
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.success(f"✅ Dataset uploaded successfully! Shape: {df.shape}")
            st.dataframe(df.head())
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
    else:
        st.info("👆 Upload a dataset to preview it here.")

# -------------------- TAB 2: MODEL TRAINING --------------------
with tab2:
    if uploaded_file is not None:
        try:
            df = read_data(uploaded_file)
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.subheader("⚙️ Configure & Train Model")
            target_column = st.selectbox("🎯 Select Target Column", df.columns)
            scaler_type = st.radio("📏 Feature Scaling", ["standard", "minmax"])

            models = {
                "Logistic Regression": LogisticRegression(max_iter=500),
                "Support Vector Machine": SVC(),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier()
            }

            selected_model_name = st.selectbox("🤖 Select Model", list(models.keys()))
            model = models[selected_model_name]
            custom_model_name = st.text_input("📝 Enter Model Name", value=selected_model_name.replace(" ", "_"))

            if st.button("🚀 Train Model"):
                with st.spinner("🔄 Training in progress..."):
                    X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)
                    trained_model = train_model(X_train, y_train, model, custom_model_name)
                    accuracy = evaluate_model(trained_model, X_test, y_test)

                st.session_state["trained_model_name"] = custom_model_name
                st.session_state["accuracy"] = accuracy
                st.session_state["X_test"] = X_test
                st.session_state["y_test"] = y_test

                st.success(f"✅ Model **{custom_model_name}** trained successfully! Accuracy: **{accuracy * 100:.2f}%**")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
    else:
        st.warning("⚠️ Please upload a dataset first in the **Dataset Preview** tab.")

# -------------------- TAB 3: RESULTS --------------------
with tab3:
    if "accuracy" in st.session_state:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("📊 Model Performance")
        st.metric(label="🎯 Accuracy", value=f"{st.session_state['accuracy'] * 100:.2f}%")

        # Accuracy Visualization
        fig, ax = plt.subplots()
        sns.barplot(x=[st.session_state["trained_model_name"]],
                    y=[st.session_state["accuracy"] * 100],
                    palette="Blues", ax=ax)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Train a model in the **Model Training** tab to view results here.")

# -------------------- TAB 4: DOWNLOAD MODELS --------------------
with tab4:
    if "trained_model_name" in st.session_state:
        model_path = os.path.join(trained_model_dir, f"{st.session_state['trained_model_name']}.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                st.download_button(
                    label="📥 Download Trained Model",
                    data=f,
                    file_name=f"{st.session_state['trained_model_name']}.pkl",
                    mime="application/octet-stream"
                )
    else:
        st.info("👆 Train a model first to enable downloads.")

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 🧠 About AutoML")
    st.info(
        "An advanced AutoML platform to:\n"
        "- 📂 Upload datasets\n"
        "- ⚙️ Preprocess automatically\n"
        "- 🤖 Train ML models\n"
        "- 📊 Evaluate accuracy\n"
        "- 📥 Download trained models"
    )
    st.markdown("---")
    st.markdown("👨‍💻 **Developed by:** Sanjan Gowda")
    st.markdown("🌐 **Version:** 3.0.0")
