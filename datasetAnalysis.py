import pandas as pd

df = pd.read_csv('Student Depression Dataset.csv')

print(df.head())


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('Student Depression Dataset.csv')






numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in ['Depression', 'id']]

print("Numerical columns:", numerical_cols)

colors = df['Depression'].map({1: 'red', 0: 'blue'})



# Plot all numerical columns vs Depression as subplots
import math

n = len(numerical_cols)
cols = 2
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    axes[i].scatter(df[col], df['Depression'], c=colors, alpha=0.7)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Depression')
    axes[i].set_title(f'{col} vs Depression')
    axes[i].grid(True)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
