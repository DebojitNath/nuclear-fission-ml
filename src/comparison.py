import pandas as pd
import matplotlib.pyplot as plt
import random
import joblib
from simulation import simulate_fission
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

def classify_run(keff):
    if keff == -1:
        return "initial"
    elif 0 <= keff < 0.95:
        return "extinguished"
    elif 0.95 <= keff <= 1.05:
        return "equilibrium"
    else:
        return "supercritical"

def compare_models(steps=20):
    # Run a single simulation
    initial_neutrons = random.randint(1,5)
    steps = 20
    multiplication_constant = random.uniform(0.85*2.5, 1.15*2.5)
    seed = random.randint(0, 10000)
    sim_results = simulate_fission(
        initial_neutrons=initial_neutrons,
        multiplication_constant=multiplication_constant,
        steps=steps,
        run_id=1,
        seed=seed
    )
    sim_df = pd.DataFrame(sim_results)

    # Ensure prev_neutrons and prev_keff exist
    sim_df["prev_neutrons"] = sim_df["neutrons"].shift(1).fillna(-1)
    sim_df["prev_keff"] = sim_df["keff"].shift(1).fillna(-1)

    # Classification column
    sim_df["classification"] = [classify_run(k) for k in sim_df["keff"]]

    # Load models
    reg_model = joblib.load("models/regression_model.pkl")
    clf_model = joblib.load("models/classification_model.pkl")
    le = joblib.load("models/clf_label_encoder.pkl")

    # Regression prediction
    sim_df["interaction"] = sim_df["prev_neutrons"] * sim_df["neutrons"]
    x_reg = sim_df[["prev_neutrons", "neutrons", "prev_keff", "keff", "interaction"]]
    sim_df["ml_neutrons_next"] = reg_model.predict(x_reg)

    # Classification prediction
    sim_df.loc[0, "ml_classification"] = "initial" 
    x_clf = sim_df.loc[1:, ["prev_neutrons", "neutrons", "prev_keff", "keff"]]
    sim_df.loc[1:, "ml_classification"] = le.inverse_transform(clf_model.predict(x_clf))

    # Regression metrics
    y_true_reg = sim_df["next_neutrons"] if "next_neutrons" in sim_df else sim_df["neutrons"]
    y_pred_reg = sim_df["ml_neutrons_next"]

    print("\n-> Regression Model Evaluation:")
    print(f"  RMSE: {root_mean_squared_error(y_true_reg, y_pred_reg):.3f}")
    print(f"  MAE:  {mean_absolute_error(y_true_reg, y_pred_reg):.3f}")
    print(f"  R²:   {r2_score(y_true_reg, y_pred_reg):.3f}")

    # Classification metrics
    y_true_clf = sim_df["classification"]
    y_pred_clf = sim_df["ml_classification"]

    print("\n-> Classification Model Evaluation:")
    print(f"  Accuracy: {accuracy_score(y_true_clf, y_pred_clf):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_clf, y_pred_clf))
    print("\nClassification Report:")
    print(classification_report(y_true_clf, y_pred_clf))

    # Plot regression comparison
    plt.figure(figsize=(10, 5))
    plt.plot(sim_df.index, y_true_reg, marker="o", label="True Neutrons")
    plt.plot(sim_df.index, y_pred_reg, marker="x", label="ML Prediction")
    plt.xlabel("Step")
    plt.ylabel("Neutrons")
    plt.title("Markov Simulation vs ML Regression Prediction")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot classification comparison
    plt.figure(figsize=(10, 5))
    label_map = {"initial": -1, "extinguished": 0, "equilibrium": 1, "supercritical": 2}
    sim_df["class_true_num"] = sim_df["classification"].map(label_map)
    sim_df["class_pred_num"] = sim_df["ml_classification"].map(label_map)
    plt.scatter(sim_df.index, sim_df["class_true_num"], marker="o", color="blue", label="True Class")
    plt.scatter(sim_df.index, sim_df["class_pred_num"], marker="x", color="red", label="Predicted Class")
    plt.yticks([-1, 0, 1, 2], ["Initial", "Extinguished", "Equilibrium", "Supercritical"])
    plt.ylabel("Classification")
    plt.title("True vs Predicted Classification")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    compare_models(steps=20)
