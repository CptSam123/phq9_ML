import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv(r'C:\Users\SAMEER\Downloads\PHQ-9 Student Depression Dataset\data.csv')  # Replace with your actual CSV filename

# Drop missing values
df = df.dropna(subset=['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9'])

# Convert all q1–q9 to int
q_cols = ['q1','q2','q3','q4','q5','q6','q7','q8','q9']
df[q_cols] = df[q_cols].astype(int)

# Calculate total score (if not present)
df['total_score'] = df[q_cols].sum(axis=1)

# Define PHQ-9 severity levels based on score
def score_to_label(score):
    if score <= 4:
        return 'Minimal'
    elif score <= 9:
        return 'Mild'
    elif score <= 14:
        return 'Moderate'
    elif score <= 19:
        return 'Moderately Severe'
    else:
        return 'Severe'

df['severity'] = df['total_score'].apply(score_to_label)

# Features and labels
X = df[q_cols]
y = df['severity'].astype('category').cat.codes

# Save label map
label_map = dict(enumerate(df['severity'].astype('category').cat.categories))
joblib.dump(label_map, 'label_map.pkl')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))

# Save model
joblib.dump(model, 'depression_model.pkl')
print("✅ Model saved as 'depression_model.pkl'")
