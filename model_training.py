import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load dataset
df = pd.read_csv("data/student-mat-8features.csv")

# Encode categorical features
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df["schoolsup"] = le.fit_transform(df["schoolsup"])
df["famsup"] = le.fit_transform(df["famsup"])
df["internet"] = le.fit_transform(df["internet"])

# Convert to Pass/Fail
df["Pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

# Features and target
X = df.drop(["G3", "Pass"], axis=1)
y = df["Pass"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

# Create models folder if not exist
if not os.path.exists("models"):
    os.makedirs("models")

# Save the trained model
with open("models/student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Trained and Saved Successfully")
