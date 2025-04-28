import pandas as pd

df = pd.read_csv('Student Depression Dataset.csv')

print(df.head())


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('Student Depression Dataset.csv')




# Numerical features

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
plt.show(block=False)


# Categorical features
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Categorical columns and number of unique categories:\n")
for col in categorical_cols:
    n_unique = df[col].nunique()
    print(f"{col}: {n_unique} categories")



import seaborn as sns


# Categorical columns
categorical_cols = [
    'Gender',
    'City',
    'Profession',
    'Sleep Duration',
    'Dietary Habits',
    'Degree',
    'Have you ever had suicidal thoughts ?',
    'Family History of Mental Illness'
]

sns.set(style="whitegrid")

fig, axes = plt.subplots(4, 2, figsize=(18, 20))  # 4 rows, 2 columns
axes = axes.flatten()  # Flatten to easily index

# Plot each categorical column
for idx, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, hue='Depression', palette={0: 'blue', 1: 'red'}, ax=axes[idx])
    axes[idx].set_title(f'{col} vs Depression')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].legend(title='Depression', labels=['No', 'Yes'])

# Hide any empty subplots (if any)
for j in range(len(categorical_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show(block=False)


# Correlation matrix for numerical features
corr = df[numerical_cols + ['Depression']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix for Numerical Features')
plt.tight_layout()
plt.show(block=False)


plt.show()
