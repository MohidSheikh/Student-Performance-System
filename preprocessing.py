import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
# Load the dataset
df = pd.read_csv('cleaned_data.csv')

#printing the first few rows of the dataset to understand its structure
df.head()

#checking for missing values in the dataset
df.isnull().sum()

features = [
    "age",
    "gender",
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "part_time_job",
    "attendance_percentage",
    "sleep_hours",
    "diet_quality",
    "exercise_frequency",
    "parental_education_level",
    "internet_quality",
    "mental_health_rating",
    "extracurricular_participation"
]
target = "exam_score"

X = df[features]
y = df[target]



#printing Numeric and categorical columns

#numeric columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
print("Numeric columns:", numeric_cols)


#categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
print("Categorical columns:", categorical_cols)



##Regression model to predict exam scores based on the features
y_reg = df["exam_score"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    random_state=42
)

reg_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

X_reg_train_processed = reg_preprocessor.fit_transform(X_reg_train)
X_reg_test_processed = reg_preprocessor.transform(X_reg_test)

joblib.dump(reg_preprocessor, "reg_preprocessor.pkl")
joblib.dump(X_reg_train_processed, "X_reg_train_processed.pkl")
joblib.dump(X_reg_test_processed, "X_reg_test_processed.pkl")
joblib.dump(y_reg_train, "y_reg_train.pkl")
joblib.dump(y_reg_test, "y_reg_test.pkl")

print("Regression preprocessing completed.")
