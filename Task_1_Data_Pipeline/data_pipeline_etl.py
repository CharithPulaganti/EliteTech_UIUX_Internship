# data_pipeline_etl.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Step 1: Load the data
data = pd.read_csv('sample_data.csv')
print("Original Data:\n", data)

# Step 2: Handle missing values (fill with mean)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Step 3: Feature Scaling
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numeric_cols])

# Step 4: Create transformed DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)

# Step 5: Save the processed data
os.makedirs('processed', exist_ok=True)
scaled_df.to_csv('processed/transformed_data.csv', index=False)

print("\nTransformed Data:\n", scaled_df)
print("\n✅ Data saved to 'processed/transformed_data.csv'")
