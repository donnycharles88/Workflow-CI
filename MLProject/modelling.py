import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import shutil
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to clean existing folder
def clean_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

# Function to create timestamped folder name
def create_timestamped_path(base_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"

# Main training function
def train_model_with_tuning(n_estimators=100, max_depth=10, dataset="processed_dataset.csv"):
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Load dataset
    file_path = dataset
    if not os.path.exists(file_path):
        print(f"❌ Dataset tidak ditemukan di path: {file_path}")
        return

    data = pd.read_csv(file_path)
    if "price" not in data.columns:
        print("❌ Kolom 'price' tidak ditemukan dalam dataset.")
        return

    # Split features and target
    X = data.drop(columns=["price"])
    y = data["price"]

    # Convert categorical columns to numeric
    categorical_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.astype('float64')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [n_estimators, n_estimators + 50, n_estimators + 100],
        'max_depth': [max_depth, max_depth + 10, max_depth + 20]
    }

    # MLflow experiment setup
    # Use relative path for tracking URI to avoid hardcoding
    mlflow.set_tracking_uri("file://./mlruns")
    mlflow.set_experiment("Computer Prices")

    # Inside train_model_with_tuning
    if mlflow.active_run():
        print("Using existing MLflow run")
    else:
        print("Starting new MLflow run")
        mlflow.start_run(run_name="RandomForest_ComputerPrice")
    with mlflow.start_run(nested=True) if mlflow.active_run() else mlflow.start_run(run_name="RandomForest_ComputerPrice"):
        print(f"Active Run ID: {mlflow.active_run().info.run_id}")


        # Manual logging of parameters
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("scoring_metric", "r2")
        mlflow.log_param("n_estimators_input", n_estimators)
        mlflow.log_param("max_depth_input", max_depth)
        mlflow.log_param("dataset", dataset)

        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Manually log best parameters
        mlflow.log_param("best_n_estimators", best_params['n_estimators'])
        mlflow.log_param("best_max_depth", best_params['max_depth'])

        # Predict and evaluate
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Manually log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)

        print(f"✅ Best Parameters: {best_params}")
        print(f"✅ MAE: {mae:.4f}")
        print(f"✅ MSE: {mse:.4f}")
        print(f"✅ R²: {r2:.4f}")

        # Save model
        model_path = create_timestamped_path("best_model_rf")
        clean_folder(model_path)
        mlflow.sklearn.save_model(sk_model=best_model, path=model_path)
        mlflow.log_artifacts(model_path, artifact_path="model")

        # Residual plot
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.6)
        ax.hlines(0, xmin=min(y_pred), xmax=max(y_pred), color='red', linestyles='--')
        ax.set_xlabel('Predicted Price')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residual Plot (R2: {r2:.4f})')

        plot_filename = create_timestamped_path("residual_plot") + ".png"
        plt.savefig(plot_filename)
        plt.close(fig)

        mlflow.log_artifact(plot_filename, artifact_path="model")

if __name__ == "__main__":
    import sys
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    dataset = sys.argv[3] if len(sys.argv) > 3 else "processed_dataset.csv"
    train_model_with_tuning(n_estimators, max_depth, dataset)