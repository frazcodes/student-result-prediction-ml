import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "StudyHours": [1,2,3,4,5,6,7,8],
    "SleepHours": [5,6,6,7,7,8,8,9],
    "Attendance": [50,60,65,70,75,80,85,90],
    "Result": [0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Features & Target
X = df[["StudyHours", "SleepHours", "Attendance"]]
y = df["Result"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (IMPORTANT)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# -------- USER INPUT --------
study = float(input("Enter Study Hours: "))
sleep = float(input("Enter Sleep Hours: "))
attendance = float(input("Enter Attendance: "))

# Convert to DataFrame (FIX WARNING)
user_data = pd.DataFrame([[study, sleep, attendance]], 
                         columns=["StudyHours", "SleepHours", "Attendance"])

# Scale user input
user_scaled = scaler.transform(user_data)

# Prediction
prediction = model.predict(user_scaled)

if prediction[0] == 1:
    print("Result: PASS ✅")
else:
    print("Result: FAIL ❌")
