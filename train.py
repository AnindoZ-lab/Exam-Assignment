import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    # 1. Setup - Load the cleaned data from Step 3
    train_df = pd.read_csv("data/processed/train.csv")
    X = train_df.drop("target", axis=1)
    y = train_df["target"]

    # 2. MLflow Tracking Block
    # This automatically stores your "logbook" in a folder called /mlruns
    with mlflow.start_run():
        # Define Hyperparameters (The "Recipe")
        n_estimators = 100
        max_depth = 5
        
        # LOG PARAMETERS
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train the Model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X, y)
        
        # LOG METRICS (The "Result")
        acc = accuracy_score(y, model.predict(X))
        mlflow.log_metric("accuracy", acc)

        # 3. LOG ARTIFACTS (The "Files")
        # Log the trained model
        mlflow.sklearn.log_model(model, "model")
        
        # Log the preprocessor from Step 3 (Crucial for reproducibility!)
        if os.path.exists("models/preprocessor.pkl"):
            mlflow.log_artifact("models/preprocessor.pkl")
            
        print(f"✅ Run Successful! Accuracy: {acc}. Artifacts stored in MLflow.")

if __name__ == "__main__":
    train_model()
