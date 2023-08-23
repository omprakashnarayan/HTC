# Importing the required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Load the cleaned data
df = pd.read_excel("cleaned_data.xlsx")


# Implementing feature engineering
# Let's create a severity score as a combination of relevant features
df['severity_score'] = df['num_lab_procedures'] + df['num_procedures'] + df['num_medications'] + df['time_in_hospital']

# Using KMeans clustering on diag_1, diag_2, diag_3 and severity_score for categorization
features_for_clustering = df[['diag_1', 'diag_2', 'diag_3', 'severity_score']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster_category'] = kmeans.fit_predict(features_for_clustering)

# Map cluster_category to 'High', 'Medium', 'Low'
def map_cluster_to_category(cluster):
    if cluster == 0:
        return 'Low'
    elif cluster == 1:
        return 'Medium'
    else:
        return 'High'

df['cluster_category'] = df['cluster_category'].apply(map_cluster_to_category)

# Calculate the counts of patients in each cluster category
cluster_counts = df['cluster_category'].value_counts()

# Print the cluster category counts
print("Cluster Category Counts:")
print(cluster_counts)

# Assign severity scores to patients
# This is a simplified example, actual scoring logic might be more complex
severity_scores = df['severity_score']

# Identify high-cost patients
high_cost_patients = df[severity_scores >= severity_scores.quantile(0.75)]

# Segregate patients based on care paths
care_intervention_patients = high_cost_patients[high_cost_patients["num_medications"] > high_cost_patients["num_medications"].median()]
lower_cost_care_patients = high_cost_patients[high_cost_patients["num_medications"] <= high_cost_patients["num_medications"].median()]

# Create a "No Care Required" category for patients not in either "Care Intervention" or "Lower Cost Care"
no_care_required_patients = df[
    ~df.index.isin(care_intervention_patients.index) &
    ~df.index.isin(lower_cost_care_patients.index)
]

# Add patients' care path categories to the DataFrame
df.loc[care_intervention_patients.index, 'care_path'] = 'Care Intervention'
df.loc[lower_cost_care_patients.index, 'care_path'] = 'Lower Cost Care'
df.loc[no_care_required_patients.index, 'care_path'] = 'No Care Required'

# Print the counts of patients in each care path
print("Care Intervention Patients:", care_intervention_patients.shape[0])
print("Lower Cost Care Patients:", lower_cost_care_patients.shape[0])
print("No Care Required Patients:", no_care_required_patients.shape[0])

# Print the dataframe with added care_path column
print(df)
df.to_excel("sorted_data.xlsx")
