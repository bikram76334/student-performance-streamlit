import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.title("Student Performance Prediction (Pass/Fail)")

# ---------------- LOAD DATASET ----------------
df = pd.read_csv("data/student-mat-8features.csv")

# Convert to Pass/Fail
df["Pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

# Encode categorical columns
le_sex = LabelEncoder()
le_support = LabelEncoder()
le_fam = LabelEncoder()
le_internet = LabelEncoder()

df["sex"] = le_sex.fit_transform(df["sex"])
df["schoolsup"] = le_support.fit_transform(df["schoolsup"])
df["famsup"] = le_fam.fit_transform(df["famsup"])
df["internet"] = le_internet.fit_transform(df["internet"])

# ---------------- ACCURACY ----------------
X = df.drop(["G3", "Pass"], axis=1)
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
model = pickle.load(open("models/student_model.pkl", "rb"))

# LIVE Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f" Model Accuracy: {acc*100:.2f}%")

# ---------------- USER INPUT ----------------
sex = st.selectbox("Sex", ["M", "F"])
studytime = st.slider("Study Time (1-4)", 1, 4, 2)
failures = st.slider("Failures (0-3)", 0, 3, 0)
absences = st.slider("Absences (0-30)", 0, 30, 3)
schoolsup = st.selectbox("School Support", ["yes", "no"])
famsup = st.selectbox("Family Support", ["yes", "no"])
internet = st.selectbox("Internet Access", ["yes", "no"])
health = st.slider("Health (1-5)", 1, 5, 3)

# Encode user inputs using same encoders
sex = le_sex.transform([sex])[0]
schoolsup = le_support.transform([schoolsup])[0]
famsup = le_fam.transform([famsup])[0]
internet = le_internet.transform([internet])[0]

# ---------------- PREDICTION ----------------
if st.button("Predict Pass/Fail"):
    data = [[sex, studytime, failures, absences, schoolsup, famsup, internet, health]]
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success(" Prediction: PASS")
    else:
        st.error(" Prediction: FAIL")

    report = f"""
    Student Report
    --------------------------
    Sex: {sex}
    Study Time: {studytime}
    Failures: {failures}
    Absences: {absences}
    School Support: {schoolsup}
    Family Support: {famsup}
    Internet: {internet}
    Health: {health}

    Result: {"PASS" if prediction[0]==1 else "FAIL"}
    """

    st.download_button(" Download Report", report, file_name="student_report.txt")

# ---------------- DASHBOARD ----------------
st.subheader("Dashboard Graphs")

# Graph 1: Pass vs Fail
fig1, ax1 = plt.subplots()
df["Pass"].value_counts().plot(kind="bar", ax=ax1)
ax1.set_xticklabels(["Fail", "Pass"], rotation=0)
st.pyplot(fig1)

# Graph 2: Study Time vs Pass Rate
fig2, ax2 = plt.subplots()
df.groupby("studytime")["Pass"].mean().plot(kind="line", marker='o', ax=ax2)
ax2.set_xlabel("Study Time")
ax2.set_ylabel("Pass Rate")
st.pyplot(fig2)

# Graph 3: Absences vs Pass Rate
fig3, ax3 = plt.subplots()
df.groupby("absences")["Pass"].mean().plot(kind="line", ax=ax3)
ax3.set_xlabel("Absences")
ax3.set_ylabel("Pass Rate")
st.pyplot(fig3)
