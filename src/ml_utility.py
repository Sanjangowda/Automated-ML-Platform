import os
import pickle
import pandas as pd
import chardet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Get the working directory of the main.py file
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)


# Step 1: Read the data (Enhanced)
import os
import io

def read_data(uploaded_file):
    """
    Reads datasets from the uploaded file via Streamlit.
    Supports CSV, Excel, TSV, and JSON formats.
    """
    if uploaded_file is None:
        raise ValueError("No file uploaded. Please upload a dataset.")

    try:
        # Get file extension
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Handle CSV files
        if file_ext == ".csv":
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)  # Reset buffer position
                raw_data = uploaded_file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(io.BytesIO(raw_data), encoding=encoding)
                except Exception:
                    df = pd.read_csv(io.BytesIO(raw_data), encoding=encoding, on_bad_lines="skip")
            except pd.errors.ParserError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")

        # Handle Excel files (.xlsx and .xls)
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        # Handle TSV files
        elif file_ext == ".tsv":
            try:
                df = pd.read_csv(uploaded_file, sep="\t", encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                raw_data = uploaded_file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                uploaded_file.seek(0)
                df = pd.read_csv(io.BytesIO(raw_data), sep="\t", encoding=encoding, on_bad_lines="skip")

        # Handle JSON files
        elif file_ext == ".json":
            df = pd.read_json(uploaded_file)

        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to read dataset: {str(e)}")


# Step 2: Preprocess the data (Improved)
def preprocess_data(df, target_column, scaler_type):
    """
    Preprocesses the dataset:
    - Handles missing values
    - Scales numeric features
    - One-hot encodes categorical features
    - Ensures unseen categories in test data won't cause errors
    """

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify column types
    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Split into train/test sets first
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle numerical columns
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

        # Scaling
        if scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        # Save scaler for future predictions
        with open(f"{parent_dir}/trained_model/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    else:
        scaler = None

    # Handle categorical columns
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
        # One-hot encoding (ignore unseen categories)
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output =False)
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])

        # Convert encoded arrays to DataFrames
        X_train_encoded = pd.DataFrame(
            X_train_encoded,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=X_train.index
        )
        X_test_encoded = pd.DataFrame(
            X_test_encoded,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=X_test.index
        )

        # Merge back
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)

        # Save encoder for predictions
        with open(f"{parent_dir}/trained_model/encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
    else:
        encoder = None

    return X_train, X_test, y_train, y_test


# Step 3: Train the model
def train_model(X_train, y_train, model, model_name):
    """
    Trains the model and saves it for future use.
    """
    model.fit(X_train, y_train)
    with open(f"{parent_dir}/trained_model/{model_name}.pkl", 'wb') as file:
        pickle.dump(model, file)
    return model


# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model and returns accuracy.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return round(accuracy, 2)


# Step 5: Predict on new data using saved encoder & scaler
def predict_new_data(model_name, new_df):
    """
    Predicts using a saved model and encoder.
    Ensures that preprocessing during prediction matches training.
    """
    # Load trained model
    with open(f"{parent_dir}/trained_model/{model_name}.pkl", 'rb') as file:
        model = pickle.load(file)

    # Load saved encoder and scaler if they exist
    encoder_path = f"{parent_dir}/trained_model/encoder.pkl"
    scaler_path = f"{parent_dir}/trained_model/scaler.pkl"

    encoder = pickle.load(open(encoder_path, 'rb')) if os.path.exists(encoder_path) else None
    scaler = pickle.load(open(scaler_path, 'rb')) if os.path.exists(scaler_path) else None

    # Handle numerical scaling
    numerical_cols = new_df.select_dtypes(include=['number']).columns
    if scaler and len(numerical_cols) > 0:
        new_df[numerical_cols] = scaler.transform(new_df[numerical_cols])

    # Handle categorical encoding
    categorical_cols = new_df.select_dtypes(include=['object', 'category']).columns
    if encoder and len(categorical_cols) > 0:
        new_encoded = encoder.transform(new_df[categorical_cols])
        new_encoded_df = pd.DataFrame(
            new_encoded,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=new_df.index
        )
        new_df = pd.concat([new_df.drop(columns=categorical_cols), new_encoded_df], axis=1)

    # Make prediction
    return model.predict(new_df)
