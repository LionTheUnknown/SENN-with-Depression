import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import layers, models
from lime import lime_tabular
import numpy as np

# Get original values for the test instance
def get_original_values(instance_idx, X_test_scaled, scaler, feature_names, df):
    X_test_original = scaler.inverse_transform(X_test_scaled)
    
    instance = X_test_original[instance_idx]
    
    original_values = {}
    for i, feature in enumerate(feature_names):
        value = instance[i]
        
        # For categorical features, convert back to original category
        if feature in cat_cols:
            unique_values = df[feature].unique()
            closest_idx = int(round(value))
            # Ensure index is in range
            closest_idx = max(0, min(closest_idx, len(unique_values)-1))
            original_values[feature] = unique_values[closest_idx]
        else:
            original_values[feature] = value
    return original_values


# Load dataset
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('Student Depression Dataset.csv').dropna()

# Drop identifier column
df = df.drop(columns=['id'])

# Categorical columns (manually specified)
cat_cols = [
    'Gender', 'Profession', 'Sleep Duration', 'City', 'Dietary Habits',
    'Degree', 'Have you ever had suicidal thoughts ?',
    'Family History of Mental Illness'
]

# Encode categorical features
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Separate features and target
X = df.drop('Depression', axis=1)
y = df['Depression']

# Save column names
feature_names = X.columns.tolist()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# Define Keras model
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")



# --- LIME Explanation ---
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=['Not Depressed', 'Depressed'],
    mode='classification'
)

i = 5  # instance index
exp = explainer.explain_instance(
    data_row=X_test[i],
    predict_fn=lambda x: np.hstack((1 - model.predict(x), model.predict(x)))
)



# Display original values for instance at index 5
original_values = get_original_values(5, X_test, scaler, feature_names, df)
print("\nOriginal values for instance at index 5:")
for feature, value in original_values.items():
    print(f"{feature}: {value}")



# exp.show_in_notebook(show_table=True, show_all=False)
os.makedirs('LIME-results', exist_ok=True)
exp.save_to_file('LIME-results/lime_explanation.html')

