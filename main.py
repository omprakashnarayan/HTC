import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
url = "https://raw.githubusercontent.com/saurabhtayal/Diabetic-Patients-Readmission-Prediction/main/diabetic_data.csv"
data = pd.read_excel(cleaneddata.xsls)

# 1. Data Cleaning
data.dropna(inplace=True)  # Remove rows with missing values

# 2. Outlier Handling (you can implement outlier detection and handling techniques here)

# 3. EDA (analyze the dataset)

# 4. Categorical Data Encoding
categorical_features = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'medical_specialty', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide_metformin', 'glipizide_metformin', 'glimepiride_pioglitazone', 'metformin_rosiglitazone', 'metformin_pioglitazone', 'change', 'diabetesMed']

# Apply label encoding to categorical features
label_encoder = LabelEncoder()
for feature in categorical_features:
    data[feature] = label_encoder.fit_transform(data[feature])

# 5. Feature Engineering (you can create new features based on domain knowledge)

# Separate features and target
X = data.drop('readmitted', axis=1)
y = data['readmitted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Machine Learning Model (Random Forest Classifier)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
