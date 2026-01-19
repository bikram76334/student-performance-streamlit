
# Student Performance Prediction (Pass/Fail)

This is a **Streamlit web application** that predicts whether a student will **pass or fail** based on their study habits and background features using a **Random Forest Classifier**.

---

## Features

- Predict **Pass/Fail** based on student data  
- Live **Model Accuracy** display  
- Dashboard graphs for data analysis  
- Download report feature (TXT format)

---

##  Project Structure

student-performance-streamlit/
│
├── app.py
├── models/
│ └── student_model.pkl
├── data/
│ └── student-mat-8features.csv
└── README.md



---

## Dataset

Dataset used: **student-mat-8features.csv**  
(Contains features like study time, failures, absences, support, health, etc.)

---

##  Installation

### 1. Clone the repository

git clone https://github.com/bikram76334/student-performance-streamlit.git
cd student-performance-streamlit

2. Create virtual environment
python -m venv venv

3. Activate virtual environment
Windows:
venv\Scripts\activate

4. Install dependencies
pip install -r requirements.txt
 Run the App

streamlit run app.py
---
5.If you want to retrain the model:

python train_model.py
---

6. Usage
Open the app in browser
---
7.Enter student details

Click Predict Pass/Fail
---
8.Download the report

Dashboard
The app shows:

Pass vs Fail chart

Study Time vs Pass Rate

Absence vs Pass Rate
---
9. Author
Bikram Chapagain
---
10. License
This project is for educational purposes.
---








