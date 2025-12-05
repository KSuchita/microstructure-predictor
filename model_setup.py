import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
df = pd.read_csv("cleaned_dataset.csv")

df['microstructure_class'] = df['composition'] + ' | ' + df['treatment']
df['microstructure_id'] = pd.factorize(df['microstructure_class'])[0]

image_map = df.drop_duplicates(subset=['microstructure_id']).set_index('microstructure_id')['image_name'].to_dict()

mlb = MultiLabelBinarizer()
lb_treatment = LabelBinarizer()

comp_ohe = mlb.fit_transform(df['composition'].fillna('').astype(str).apply(lambda x: x.split()))
treat_ohe = lb_treatment.fit_transform(df['treatment'])

X = pd.concat([
    pd.DataFrame(comp_ohe, columns=[f'comp_{c}' for c in mlb.classes_]),
    pd.DataFrame(treat_ohe, columns=[f'treat_{c}' for c in lb_treatment.classes_])
], axis=1)

y = df['microstructure_id']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

dump(model, 'model.joblib')
dump(mlb, 'mlb_composition.joblib')
dump(lb_treatment, 'lb_treatment.joblib')
dump(image_map, 'image_map.joblib')
