from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import os

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

model_dir = "models"  # Remove "../" to store in the current project directory
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "random_forest.joblib")
# Save model
dump(model, model_path)
print("Model saved successfully!")