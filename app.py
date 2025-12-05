from flask import Flask, render_template, request, jsonify
from joblib import load
from PIL import Image
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# CONSTANTS
IMAGE_DIR = 'static/Resized'
OUTPUT_DIR = 'static'
BLENDED_IMAGE_FILENAME = 'blended_output.png'
BLENDED_IMAGE_PATH = os.path.join(OUTPUT_DIR, BLENDED_IMAGE_FILENAME)

# LOAD MODEL + ENCODERS + MAP
try:
    model = load('model.joblib')
    mlb_composition = load('mlb_composition.joblib')
    lb_treatment = load('lb_treatment.joblib')
    image_map = load('image_map.joblib')
    print("All components loaded successfully.")
except Exception as e:
    print(f"Error loading components: {e}")

# ------------ 2. PREPROCESS INPUT -------------
def preprocess_input(comp_input, treat_input):
    comp_tokens = comp_input.split()
    comp_ohe = mlb_composition.transform([comp_tokens])

    treat_array = np.array([treat_input])
    treat_ohe = lb_treatment.transform(treat_array)

    # fix shape issues
    if treat_ohe.ndim == 1:
        treat_ohe = treat_ohe.reshape(1, -1)

    X_pred = np.hstack([comp_ohe, treat_ohe])
    return X_pred

# ------------ 3. IMAGE BLENDING --------------
def blend_images(predictions):
    if not predictions:
        return None

    try:
        first_img_path = os.path.join(IMAGE_DIR, predictions[0]['image_name'])
        ref_img = Image.open(first_img_path).convert('RGB')
        target_size = ref_img.size

        blended = np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)

        total_prob = sum(p['probability'] for p in predictions)

        for p in predictions:
            img_path = os.path.join(IMAGE_DIR, p['image_name'])
            img = Image.open(img_path).convert('RGB').resize(target_size)
            img_array = np.array(img, dtype=np.float32)

            weight = p['probability'] / total_prob
            blended += img_array * weight

        blended = np.clip(blended, 0, 255).astype(np.uint8)
        blended_img = Image.fromarray(blended)
        blended_img.save(BLENDED_IMAGE_PATH)

        return BLENDED_IMAGE_FILENAME

    except Exception as e:
        print(f"Error in blending: {e}")
        return None

# ------------ 4. FLASK ROUTES -------------------

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    comp_input = request.form.get('composition')
    treat_input = request.form.get('treatment')

    if not comp_input or not treat_input:
        return jsonify({'error': 'Please provide values for both Composition and Treatment.'})

    try:
        X_pred = preprocess_input(comp_input, treat_input)
        probabilities = model.predict_proba(X_pred)[0]

        top_indices = np.argsort(probabilities)[::-1][:3]

        predictions = []
        for idx in top_indices:
            prob = probabilities[idx]
            if prob < 0.01:
                continue

            micro_id = model.classes_[idx]
            image_name = image_map.get(micro_id, 'image_not_found.png')

            predictions.append({
                'probability': float(prob),
                'image_name': image_name
            })

        if not predictions:
            return jsonify({'error': 'Prediction confidence too low.'})

        blended_filename = blend_images(predictions)

        if blended_filename is None:
            return jsonify({'error': 'Blending failed.'})

        return jsonify({
            'status': 'success',
            'composition_input': comp_input,
            'treatment_input': treat_input,
            'image_src': blended_filename,
            'raw_predictions': predictions
        })

    except Exception as e:
        return jsonify({'error': f"Prediction error: {e}"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
