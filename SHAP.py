import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import layers, models
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('Student Depression Dataset.csv')

df = df.dropna()

cat_cols = ['Gender', 'Profession', 'Sleep Duration', 'City', 'Dietary Habits', 'Degree','Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])


X = df.drop('Depression', axis=1) 
y = df['Depression']  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Subject to change
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")


X_sample = X_test[:100]  


explainer = shap.KernelExplainer(model.predict, X_sample)

shap_values = explainer.shap_values(X_sample)


shap.summary_plot(shap_values, X_sample)