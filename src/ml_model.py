import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

dataset_file = "data/ml_dataset.csv"

def train_markov_model():
    if not os.path.exists(dataset_file):
        print(f"[ERROR] Dataset not found: {dataset_file}")
        return
    
    df = pd.read_csv(dataset_file)

    # Regression features
    df["prev_neutrons"] = df["neutrons"].shift(1).fillna(0)
    df["prev_keff"] = df["keff"].shift(1).fillna(0)
    df["interaction"] = df["prev_neutrons"] * df["neutrons"]

    x_reg = df[["prev_neutrons", "neutrons", "prev_keff", "keff", "interaction"]]
    y_reg = df["next_neutrons"]

    # Train/test split
    split = int(0.8 * len(df))
    x_train_reg, x_test_reg = x_reg[:split], x_reg[split:]
    y_train_reg, y_test_reg = y_reg[:split], y_reg[split:]

    # Regression model
    reg_model = RandomForestRegressor(n_estimators=200, random_state=42)
    reg_model.fit(x_train_reg, y_train_reg)
    y_pred_reg = reg_model.predict(x_test_reg)

    print("\nMarkov Chain Regression Results:")
    print(f"  MAE  = {mean_absolute_error(y_test_reg, y_pred_reg):.4f}")
    print(f"  RMSE = {np.sqrt(root_mean_squared_error(y_test_reg, y_pred_reg)):.4f}")
    print(f"  R²   = {r2_score(y_test_reg, y_pred_reg):.4f}")

    joblib.dump(reg_model, "models/regression_model.pkl")

    # Classification features
    x_clf = df[["prev_neutrons", "neutrons", "prev_keff", "keff"]]
    y_clf = df["classification"]
    le = LabelEncoder()
    y_clf_encoded = le.fit_transform(y_clf)

    clf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_model.fit(x_clf, y_clf_encoded)
    y_pred_clf_encoded = clf_model.predict(x_clf)
    y_pred_clf = le.inverse_transform(y_pred_clf_encoded)

    print("\nClassification Model Evaluation:")
    print(f"  Accuracy: {accuracy_score(y_clf, y_pred_clf):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_clf, y_pred_clf))
    print("\nClassification Report:")
    print(classification_report(y_clf, y_pred_clf))

    joblib.dump(clf_model, "models/classification_model.pkl")
    joblib.dump(le, "models/clf_label_encoder.pkl")

    return reg_model, clf_model

if __name__ == "__main__":
    train_markov_model()
