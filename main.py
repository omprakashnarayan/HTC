# Importing the required library's
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error


# Load the cleaned data
df = pd.read_excel("cleaned_data.xlsx")


#Implementing feature engineering
# Let's create a severity score as a combination of relevant features
df['severity_score'] = df['num_lab_procedures'] + df['num_procedures'] + df['num_medications'] + df['time_in_hospital']

# Using RandomForestRegressor for predicting severity scores
# Defining the feature columns and target column
X = df.drop("severity_score", axis=1)  # Features
y = df["severity_score"]  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the regressor
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Categorize patient data based on predicted severity scores
def categorize_severity(score):
    if score >= 100:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

df['predicted_severity_level'] = df['severity_score'].apply(categorize_severity)
print(df[['severity_score', 'predicted_severity_level']])
# Assign severity scores to patients
# This is a simplified example, actual scoring logic might be more complex
# Identify high-cost patients
high_cost_patients = df[severity_score >= severity_score.quantile(0.75)]

# Segregate patients based on care paths
care_intervention_patients = high_cost_patients[high_cost_patients["num_medications"] > high_cost_patients["num_medications"].median()]
lower_cost_care_patients = high_cost_patients[high_cost_patients["num_medications"] <= high_cost_patients["num_medications"].median()]

# Print the counts of patients in each care path
print("Care Intervention Patients:", care_intervention_patients.shape[0])
print("Lower Cost Care Patients:", lower_cost_care_patients.shape[0])

