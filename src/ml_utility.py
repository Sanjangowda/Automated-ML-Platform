import os
import pickle
import pandas as pd
import chardet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from io import BytesIO
import io
import matplotlib.pyplot as plt

# ---------------------------
# Directory Setup
# ---------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

# ---------------------------
# Read Data
# ---------------------------
def read_data(file):
    file_ext = os.path.splitext(file.name)[1].lower()
    raw_data = file.read()
    file.seek(0)

    detected_encoding = chardet.detect(raw_data)["encoding"] or "utf-8"

    if file_ext == ".csv":
        return pd.read_csv(BytesIO(raw_data), encoding=detected_encoding, on_bad_lines="skip")
    elif file_ext in [".xlsx", ".xls"]:
        return pd.read_excel(BytesIO(raw_data), engine="openpyxl")
    elif file_ext == ".tsv":
        return pd.read_csv(BytesIO(raw_data), sep="\t", encoding=detected_encoding, on_bad_lines="skip")
    elif file_ext == ".json":
        return pd.read_json(BytesIO(raw_data))
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

# ---------------------------
# Detect Problem Type
# ---------------------------
def detect_problem_type(y):
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        return "regression"
    return "classification"

# ---------------------------
# Auto Model Selection
# ---------------------------
def auto_select_model(y):
    problem_type = detect_problem_type(y)

    if problem_type == "regression":
        return (
            RandomForestRegressor(n_estimators=200, random_state=42),
            "random_forest_regressor",
            problem_type
        )

    return (
        RandomForestClassifier(n_estimators=200, random_state=42),
        "random_forest_classifier",
        problem_type
    )

# ---------------------------
# Preprocess Data
# ---------------------------
def preprocess_data(df, target_column, scaler_type):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle datetime columns
    for col in X.select_dtypes(include=["datetime64[ns]"]).columns:
        X[col + "_year"] = X[col].dt.year
        X[col + "_month"] = X[col].dt.month
        X[col + "_day"] = X[col].dt.day
        X = X.drop(columns=[col])

    numerical_cols = X.select_dtypes(include=["number"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Numerical
    if len(numerical_cols) > 0:
        imputer = SimpleImputer(strategy="mean")
        X_train[numerical_cols] = imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = imputer.transform(X_test[numerical_cols])

        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        pickle.dump(scaler, open(f"{parent_dir}/trained_model/scaler.pkl", "wb"))

    # Categorical
    if len(categorical_cols) > 0:
        imputer = SimpleImputer(strategy="most_frequent")
        X_train[categorical_cols] = imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = imputer.transform(X_test[categorical_cols])

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_train_enc = encoder.fit_transform(X_train[categorical_cols])
        X_test_enc = encoder.transform(X_test[categorical_cols])

        X_train = pd.concat(
            [X_train.drop(columns=categorical_cols),
             pd.DataFrame(X_train_enc, columns=encoder.get_feature_names_out())],
            axis=1
        )
        X_test = pd.concat(
            [X_test.drop(columns=categorical_cols),
             pd.DataFrame(X_test_enc, columns=encoder.get_feature_names_out())],
            axis=1
        )

        pickle.dump(encoder, open(f"{parent_dir}/trained_model/encoder.pkl", "wb"))

    return X_train, X_test, y_train, y_test

# ---------------------------
# Train Model
# ---------------------------
def train_model(X_train, y_train, model=None, model_name=None):
    # 🔐 FORCE correct model based on target type
    problem_type = detect_problem_type(y_train)

    if problem_type == "regression":
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
        model_name = model_name or "random_forest_regressor"

    else:
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
        model_name = model_name or "random_forest_classifier"

    # ✅ ALIGN + NaN SAFETY
    X_train = X_train.loc[y_train.index].fillna(0)

    model.fit(X_train, y_train)

    with open(f"{parent_dir}/trained_model/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

    return model

# ---------------------------
# Evaluate Model
# ---------------------------
def evaluate_model(model, X_test, y_test):
    X_test = X_test.loc[y_test.index].fillna(0)
    y_pred = model.predict(X_test)
    problem_type = detect_problem_type(y_test)

    # -------- Regression --------
    if problem_type == "regression":
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        insight = f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}"

        return round(r2 * 100, 2), results, insight, buf


    # -------- Classification --------
    acc = accuracy_score(y_test, y_pred)

    results = pd.DataFrame({
        "Actual": y_test.astype(str),
        "Predicted": y_pred.astype(str)
    })

    dist = results["Predicted"].value_counts(normalize=True) * 100

    fig, ax = plt.subplots()
    dist.plot(kind="bar", ax=ax)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Prediction Distribution")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return round(acc * 100, 2), results, dist.to_dict(), buf

# ---------------------------
# Predict New Data
# ---------------------------
def predict_new_data(model_name, new_df):
    model = pickle.load(open(f"{parent_dir}/trained_model/{model_name}.pkl", "rb"))

    if os.path.exists(f"{parent_dir}/trained_model/scaler.pkl"):
        scaler = pickle.load(open(f"{parent_dir}/trained_model/scaler.pkl", "rb"))
        num_cols = new_df.select_dtypes(include=["number"]).columns
        new_df[num_cols] = scaler.transform(new_df[num_cols])

    if os.path.exists(f"{parent_dir}/trained_model/encoder.pkl"):
        encoder = pickle.load(open(f"{parent_dir}/trained_model/encoder.pkl", "rb"))
        cat_cols = new_df.select_dtypes(include=["object", "category"]).columns
        encoded = encoder.transform(new_df[cat_cols])
        new_df = pd.concat(
            [new_df.drop(columns=cat_cols),
             pd.DataFrame(encoded, columns=encoder.get_feature_names_out())],
            axis=1
        )

    new_df = new_df.fillna(0)
    return model.predict(new_df)
