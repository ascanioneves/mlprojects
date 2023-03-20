import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the diabetes dataset
abalone_df = pd.read_csv('abalone_dataset.csv')

# Compute the correlation matrix
corr_matrix = abalone_df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# save the figure
plt.savefig('correlation_matrix.png')