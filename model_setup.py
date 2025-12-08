import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import numpy as np
import re

# --- Normalization Function for Consistent Keys ---
def normalize_string(text):
    """Normalizes input string, ensuring numerical strings like magnification are consistent."""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    
    # Check if the text is a number and convert it to a consistent integer string ('280' instead of '280.0')
    try:
        if text.replace('.', '', 1).isdigit():
            # If it's a number, convert to float then to int to remove trailing .0, then back to string
            return str(int(float(text)))
    except ValueError:
        pass
        
    text = text.replace('-', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text
# --------------------------------------------------

# --- 1. Load Data ---
df = pd.read_csv("cleaned_dataset.csv")

# Create normalized columns for robust mapping and feature creation
df['composition_normalized'] = df['composition'].apply(normalize_string)
df['treatment_normalized'] = df['treatment'].apply(normalize_string)
# Use the same normalization logic for magnification as we will use for input
df['magnification_str'] = df['magnification'].astype(str).apply(normalize_string) 

# --- 2. Create Target Variable (Microstructure ID) ---
# Use the normalized keys for classes (Model training target)
df['microstructure_class'] = df['composition_normalized'] + ' | ' + df['treatment_normalized']
df['microstructure_id'] = pd.factorize(df['microstructure_class'])[0]

# Create a mapping from ID back to a single sample image name (for model retrieval)
image_map = df.drop_duplicates(subset=['microstructure_id']).set_index('microstructure_id')['image_name'].to_dict()

# --- NEW MAPS FOR BLENDING ---
# Tier 1 Map: Full Match (Composition + Treatment + Magnification) -> List of ALL images
df['microstructure_magnification_class'] = df['microstructure_class'] + ' | ' + df['magnification_str']
mag_class_image_list_map = df.groupby('microstructure_magnification_class')['image_name'].apply(list).to_dict()

# Tier 2 Map: Microstructure Class Match (Composition + Treatment) -> List of ALL images
class_image_list_map = df.groupby('microstructure_class')['image_name'].apply(list).to_dict()

# Tier 3 Map: Composition Only -> List of ALL images
comp_image_list_map = df.groupby('composition_normalized')['image_name'].apply(list).to_dict()
# -----------------------------

# --- 3. Feature Preparation (X) ---

# Define the encoders
mlb = MultiLabelBinarizer()
lb_treatment = LabelBinarizer()

# Use the normalized columns for training features
comp_ohe = mlb.fit_transform(df['composition_normalized'].fillna('').apply(lambda x: x.split()))
comp_ohe_df = pd.DataFrame(comp_ohe, columns=[f'comp_{c}' for c in mlb.classes_])

treatment_clean = df['treatment_normalized']
treat_ohe = lb_treatment.fit_transform(treatment_clean)
treat_ohe_df = pd.DataFrame(treat_ohe, columns=[f'treat_{c}' for c in lb_treatment.classes_])

# Combine all features (X)
X = pd.concat([comp_ohe_df, treat_ohe_df], axis=1)
y = df['microstructure_id']

# --- 4. Model Training ---
print(f"Training on {X.shape[0]} samples with {X.shape[1]} features and {len(y.unique())} microstructure classes.")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("Model training complete.")

# --- 5. Save Components ---
dump(model, 'model.joblib')
dump(mlb, 'mlb_composition.joblib')
dump(lb_treatment, 'lb_treatment.joblib')
dump(image_map, 'image_map.joblib')
dump(mag_class_image_list_map, 'mag_class_image_list_map.joblib') 
dump(class_image_list_map, 'class_image_list_map.joblib') 
dump(comp_image_list_map, 'comp_image_list_map.joblib') 

print("Model and preprocessing components saved successfully. Run this script first!")