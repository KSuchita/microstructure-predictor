import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# --------------------------------------------------
# Normalization Function (MUST MATCH app.py exactly)
# --------------------------------------------------
def normalize_string(text):
    if pd.isna(text) or text is None:
        return ""

    text = str(text).lower().strip()

    try:
        if text.replace('.', '', 1).isdigit():
            return str(int(float(text)))
    except ValueError:
        pass

    text = text.replace('-', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------
# 1. Load Dataset
# -----------------
df = pd.read_csv("cleaned_dataset.csv")

print(f"Total samples: {len(df)}")

# Drop rows with critical missing values
df = df.dropna(subset=['composition', 'treatment', 'image_name'])

print(f"Samples after cleaning: {len(df)}")

# -----------------
# 2. Normalize Columns
# -----------------
df['composition_normalized'] = df['composition'].apply(normalize_string)
df['treatment_normalized'] = df['treatment'].apply(normalize_string)

df['magnification_str'] = (
    df['magnification']
    .astype(str)
    .apply(normalize_string)
)

# -----------------
# 3. Create Classes
# -----------------
df['microstructure_class'] = (
    df['composition_normalized'] + ' | ' + df['treatment_normalized']
)

df['microstructure_id'] = pd.factorize(df['microstructure_class'])[0]

print(f"Total classes: {df['microstructure_id'].nunique()}")

# -----------------
# 4. Image Maps (FOR FLASK APP)
# -----------------
image_map = (
    df.drop_duplicates('microstructure_id')
      .set_index('microstructure_id')['image_name']
      .to_dict()
)

# Tier 1: Composition + Treatment + Magnification
df['microstructure_magnification_class'] = (
    df['microstructure_class'] + ' | ' + df['magnification_str']
)

mag_class_image_list_map = (
    df.groupby('microstructure_magnification_class')['image_name']
      .apply(list)
      .to_dict()
)

# Tier 2: Composition + Treatment
class_image_list_map = (
    df.groupby('microstructure_class')['image_name']
      .apply(list)
      .to_dict()
)

# Tier 3: Composition only
comp_image_list_map = (
    df.groupby('composition_normalized')['image_name']
      .apply(list)
      .to_dict()
)

# -----------------
# 5. Feature Engineering
# -----------------
mlb_composition = MultiLabelBinarizer()
lb_treatment = LabelBinarizer()

# Composition → Multi-hot encoding
comp_features = mlb_composition.fit_transform(
    df['composition_normalized'].apply(lambda x: x.split())
)

# Treatment → One-hot encoding
treat_features = lb_treatment.fit_transform(
    df['treatment_normalized']
)

# Combine features
X = np.hstack([comp_features, treat_features])
y = df['microstructure_id'].values

print(f"Total features: {X.shape[1]}")

# -----------------
# 6. Train Model
# -----------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

print("Model training complete.")

# -----------------
# 7. Save Everything
# -----------------
dump(model, "model.joblib")
dump(mlb_composition, "mlb_composition.joblib")
dump(lb_treatment, "lb_treatment.joblib")

dump(image_map, "image_map.joblib")
dump(mag_class_image_list_map, "mag_class_image_list_map.joblib")
dump(class_image_list_map, "class_image_list_map.joblib")
dump(comp_image_list_map, "comp_image_list_map.joblib")

print("All files saved successfully!")
print("Now run: python app.py")
