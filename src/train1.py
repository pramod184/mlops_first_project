import optuna
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment
mlflow.set_experiment("iris_optuna_experiment")

def objective(trial):
    with mlflow.start_run():
        # Suggest hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 50, 200, step=10)
        max_depth = trial.suggest_int("max_depth", 3, 20)

        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters and metrics to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        # Save best model
        # if trial.number == 0 or accuracy > objective.best_accuracy:
        #     objective.best_accuracy = accuracy
        #     dump(model, "../models/best_random_forest.joblib")
        #     mlflow.sklearn.log_model(model, "best_random_forest_model")

        return accuracy

# Track best accuracy globally
objective.best_accuracy = 0.0

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)
print("Best accuracy:", study.best_value)