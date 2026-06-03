import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import accuracy_score, r2_score

# Load the cleaned student dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Encode categorical text features into numerical values
label_encoders = {}
categorical_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 
                    'internet_quality', 'extracurricular_participation']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Define binary classification target (Pass/Fail) based on a 50% passing threshold
df['pass_fail'] = np.where(df['exam_score'] >= 50, 1, 0)

# Separate input features (X) and target outputs (y)
X = df.drop(columns=['exam_score', 'pass_fail'])  
y_score = df['exam_score']  
y_class = df['pass_fail']   

# Split dataset into training (80%) and testing (20%) subsets
X_train, X_test, y_train_score, y_test_score = train_test_split(X, y_score, test_size=0.2, random_state=42)
_, _, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Train Model 1: Random Forest Classifier for Pass/Fail Risk Assessment
clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_train, y_train_class)
acc = accuracy_score(y_test_class, clf_model.predict(X_test))
print(f"Classification Model Accuracy: {acc * 100:.2f}%")

# Train Model 2: Linear Regression for Exact Exam Score Prediction
reg_model = LinearRegression()
reg_model.fit(X_train, y_train_score)
r2 = r2_score(y_test_score, reg_model.predict(X_test))
print(f"Regression Model R2 Score: {r2 * 100:.2f}%")

# Save trained models and encoders to the serialized pickle files
with open('models/classifier_model.pkl', 'wb') as f:
    pickle.dump(clf_model, f)

with open('models/regression_model.pkl', 'wb') as f:
    pickle.dump(reg_model, f)

with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Artifacts successfully saved in 'models/' directory.\n")