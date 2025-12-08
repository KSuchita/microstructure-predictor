from flask import Flask, render_template, request, jsonify
from joblib import load
from PIL import Image
import numpy as np
import os
import re
import traceback

# ----------------- 1. Flask App Setup -----------------
app = Flask(__name__)

# Define constants
IMAGE_DIR = 'static/Resized'
OUTPUT_DIR = 'static'
BLENDED_IMAGE_FILENAME = 'blended_output.png'
BLENDED_IMAGE_PATH = os.path.join(OUTPUT_DIR, BLENDED_IMAGE_FILENAME)

# ----------------- Helper Functions -----------------

def normalize_string(text):
    if text is None or str(text).lower() == "nan" or str(text).strip() == "":
        return "" 
    
    text = str(text).lower()

    try:
        if text.replace('.', '', 1).isdigit():
            return str(int(float(text)))
    except ValueError:
        pass
        
    text = text.replace('-', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------- 2. Load Model and Components -----------------
try:
    model = load('model.joblib')
    mlb_composition = load('mlb_composition.joblib')
    lb_treatment = load('lb_treatment.joblib')
    image_map = load('image_map.joblib')
    comp_image_list_map = load('comp_image_list_map.joblib')
    class_image_list_map = load('class_image_list_map.joblib')
    mag_class_image_list_map = load('mag_class_image_list_map.joblib')
    
    print("Model and components loaded successfully.")

except Exception as e:
    print(f"Error loading components: {e}.")

# ----------------- 3. Preprocessing Function for New Input -----------------
def preprocess_input(comp_input, treat_input):
    comp_normalized = normalize_string(comp_input)
    treat_normalized = normalize_string(treat_input)
    
    comp_list = [comp_normalized.split()]
    comp_ohe = mlb_composition.transform(comp_list)
    
    treat_array = np.array([treat_normalized])
    treat_ohe = lb_treatment.transform(treat_array)
    
    X_pred = np.hstack([comp_ohe, treat_ohe])
    return X_pred

# ----------------- 4. Image Blending Function + CROPPING -----------------
def blend_images(images_to_blend):

    if not images_to_blend:
        return None, []

    first_img_path = os.path.join(IMAGE_DIR, images_to_blend[0]['image_name'])
    try:
        ref_img = Image.open(first_img_path).convert('RGB')
        target_size = ref_img.size
        
        blended_array = np.zeros((*target_size[::-1], 3), dtype=np.float32)
        weight_map = np.zeros(target_size[::-1], dtype=np.float32)
        
        total_weight_check = sum(p['weight'] for p in images_to_blend)
        used_images_summary = []

        for p in images_to_blend:
            img_path = os.path.join(IMAGE_DIR, p['image_name'])
            weight = p['weight']
            
            img = Image.open(img_path).convert('RGB').resize(target_size)
            img_array = np.array(img, dtype=np.float32)
            
            # --- ROBUST MASK DETECTION ---
            BRIGHTNESS_THRESHOLD = 20
            VARIATION_THRESHOLD = 5

            is_near_black = (img_array.sum(axis=2) < BRIGHTNESS_THRESHOLD)
            max_channel = img_array.max(axis=2)
            min_channel = img_array.min(axis=2)
            is_low_variation = (max_channel - min_channel < VARIATION_THRESHOLD)

            is_padding = np.logical_and(is_near_black, is_low_variation)
            mask_2d = np.logical_not(is_padding)
            mask_3d = mask_2d[:, :, np.newaxis]
            # --------------------------

            blended_array += (img_array * weight) * mask_3d
            weight_map += weight * mask_2d
            
            weight_percent = round((weight / total_weight_check) * 100, 2) if total_weight_check > 0 else 0.0
            used_images_summary.append({'image_name': p['image_name'], 'weight_percent': weight_percent})

        final_blended_array = np.divide(
            blended_array,
            weight_map[:, :, np.newaxis],
            out=np.zeros_like(blended_array),
            where=weight_map[:, :, np.newaxis] != 0
        )
        
        final_blended_array = np.clip(final_blended_array, 0, 255).astype(np.uint8)

        # -----------------------------
        # REMOVE PADDING COMPLETELY
        # -----------------------------
        mask_valid = weight_map > 0

        rows = np.any(mask_valid, axis=1)
        cols = np.any(mask_valid, axis=0)

        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            cropped_array = final_blended_array[rmin:rmax+1, cmin:cmax+1, :]
            blended_img = Image.fromarray(cropped_array)

        else:
            blended_img = Image.fromarray(final_blended_array)

        blended_img.save(BLENDED_IMAGE_PATH)
        
        return BLENDED_IMAGE_FILENAME, used_images_summary

    except Exception as e:
        print(f"Error during image blending: {e}")
        traceback.print_exc()
        return None, []

# ----------------- 5. Four-Tier Blending Strategy -----------------

def get_images_for_blending(comp_input, treat_input, mag_input):
    
    comp_norm = normalize_string(comp_input)
    treat_norm = normalize_string(treat_input)
    mag_norm = normalize_string(mag_input)
    
    # TIER 1
    mag_class_key = comp_norm + ' | ' + treat_norm + ' | ' + mag_norm
    if mag_class_key in mag_class_image_list_map:
        image_list = mag_class_image_list_map[mag_class_key]
        if image_list:
            weight = 1.0 / len(image_list)
            return [{'image_name': img, 'weight': weight} for img in image_list], \
                   "Tier 1: Full Match Blend"

    # TIER 2
    class_key = comp_norm + ' | ' + treat_norm
    if class_key in class_image_list_map:
        image_list = class_image_list_map[class_key]
        if image_list:
            weight = 1.0 / len(image_list)
            return [{'image_name': img, 'weight': weight} for img in image_list], \
                   "Tier 2: Class Match Blend"

    # TIER 3
    if comp_norm in comp_image_list_map:
        image_list = comp_image_list_map[comp_norm]
        if image_list:
            weight = 1.0 / len(image_list)
            return [{'image_name': img, 'weight': weight} for img in image_list], \
                   "Tier 3: Composition Match Blend"

    # TIER 4 (MODEL)
    X_pred = preprocess_input(comp_input, treat_input)
    probabilities = model.predict_proba(X_pred)[0]
    top_n_indices = np.argsort(probabilities)[::-1][:3] 
    
    images_to_blend = []
    total_prob = sum(probabilities[index] for index in top_n_indices if probabilities[index] >= 0.01)

    for index in top_n_indices:
        probability = probabilities[index]
        if probability < 0.01:
            continue

        microstructure_id = model.classes_[index]
        predicted_image_name = image_map.get(microstructure_id, None)
        
        if predicted_image_name:
            normalized_weight = probability / total_prob if total_prob > 0 else 0.0
            
            images_to_blend.append({
                'image_name': predicted_image_name, 
                'weight': normalized_weight
            })

    return images_to_blend, "Tier 4: Model-Based Blend"

# ----------------- 6. Flask Routes -----------------

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comp_input = request.form.get('composition')
    treat_input = request.form.get('treatment')
    mag_input = request.form.get('magnification')

    if not comp_input or not treat_input:
        return jsonify({'error': 'Please provide values for Composition and Treatment.'})
    
    try:
        images_to_blend, strategy = get_images_for_blending(comp_input, treat_input, mag_input)
        
        if not images_to_blend:
            return jsonify({'error': 'No relevant images found or prediction confidence too low.'})
            
        blended_filename, used_images_summary = blend_images(images_to_blend)
        
        if blended_filename:
            response = {
                'status': 'success',
                'composition_input': comp_input,
                'treatment_input': treat_input,
                'magnification_input': mag_input,
                'image_src': blended_filename,
                'strategy': strategy,
                'used_images': used_images_summary
            }
        else:
            response = {'error': 'Failed to generate blended image.'}
        
        return jsonify(response)

    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return jsonify({'error': f"Unexpected error: {e}"})

if __name__ == '__main__':
    if not os.path.exists(IMAGE_DIR):
        print(f"Warning: Directory '{IMAGE_DIR}' not found.")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        
    app.run(debug=True, host='0.0.0.0', port=5000)
