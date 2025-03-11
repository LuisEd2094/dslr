import pandas as pd

# Load dataset
data = pd.read_csv('dataset_train.csv')

# Select only numeric columns
numeric_data = data.select_dtypes(include=['number'])

# Get describe() output for numeric columns
desc = numeric_data.describe()

# Add extra statistics
desc.loc['median'] = numeric_data.median()  # Median
desc.loc['skew'] = numeric_data.skew()  # Skewness
desc.loc['kurtosis'] = numeric_data.kurtosis()  # Kurtosis
desc.loc['var'] = numeric_data.var()  # Variance
desc.loc['range'] = numeric_data.max() - numeric_data.min()  # Range

# Print result
#print(desc)

print(data.describe())

import matplotlib.pyplot as plt

data.hist(figsize=(20,20))
plt.savefig('pandas_histogram.png')
