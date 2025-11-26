import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import numpy as np

# --- 1. Load Data ---
df = pd.read_csv("cleaned_dataset.csv")

# --- 2. Create Target Variable (Microstructure ID) ---
# Combine composition and treatment to define a unique microstructure class
df['microstructure_class'] = df['composition'] + ' | ' + df['treatment']
# Factorize converts these unique strings into unique integer IDs (Y)
df['microstructure_id'] = pd.factorize(df['microstructure_class'])[0]

# Create a mapping from ID back to a sample image name (for retrieval)
image_map = df.drop_duplicates(subset=['microstructure_id']).set_index('microstructure_id')['image_name'].to_dict()

# --- 3. Feature Preparation (X) ---

# Define the encoders
mlb = MultiLabelBinarizer()
lb_treatment = LabelBinarizer()

# a) Composition (MultiLabelBinarizer for space-separated elements)
comp_ohe = mlb.fit_transform(df['composition'].fillna('').astype(str).apply(lambda x: x.split()))
comp_ohe_df = pd.DataFrame(comp_ohe, columns=[f'comp_{c}' for c in mlb.classes_])

# b) Treatment (LabelBinarizer for the categorical treatment string)
treat_ohe = lb_treatment.fit_transform(df['treatment'])
treat_ohe_df = pd.DataFrame(treat_ohe, columns=[f'treat_{c}' for c in lb_treatment.classes_])

# Combine all features (X)
X = pd.concat([comp_ohe_df, treat_ohe_df], axis=1)
y = df['microstructure_id']

# --- 4. Model Training ---
print(f"Training on {X.shape[0]} samples with {X.shape[1]} features and {len(y.unique())} microstructure classes.")

# Random Forest is used here for its robust probability prediction capabilities
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("Model training complete.")

# --- 5. Save Components ---
dump(model, 'model.joblib')
dump(mlb, 'mlb_composition.joblib')
dump(lb_treatment, 'lb_treatment.joblib')
dump(image_map, 'image_map.joblib')

print("Model and preprocessing components saved successfully. You can now run app.py.")