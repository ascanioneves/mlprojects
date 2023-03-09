import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the diabetes dataset
diabetes_df = pd.read_csv('diabetes_dataset.csv')

# Compute the correlation matrix
corr_matrix = diabetes_df.corr()
corr_with_target = corr_matrix['Outcome'].sort_values(ascending=False)

corr_pairs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
corr_pairs = corr_pairs[corr_pairs != 1]

print(corr_pairs)
