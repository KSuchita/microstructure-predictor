from flask import Flask, render_template, request, jsonify
from joblib import load
from PIL import Image
import numpy as np
import os
import re
import traceback

# -----------------
# 1. Flask App Setup
# -----------------
app = Flask(__name__)

IMAGE_DIR = 'static/Resized'
OUTPUT_DIR = 'static'
BLENDED_IMAGE_FILENAME = 'blended_output.png'
BLENDED_IMAGE_PATH = os.path.join(OUTPUT_DIR, BLENDED_IMAGE_FILENAME)

# -----------------
# Helper Functions
# -----------------
def normalize_string(text):
    if text is None or str(text).lower() == "nan" or str(text).strip() == "":
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
# 2. Load Model and Components
# -----------------
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
    print("Error loading components:", e)
    raise

# -----------------
# 3. Preprocessing Function
# -----------------
def preprocess_input(comp_input, treat_input):
    comp_norm = normalize_string(comp_input)
    treat_norm = normalize_string(treat_input)

    comp_list = [comp_norm.split()]
    X_comp = mlb_composition.transform(comp_list)

    X_treat = lb_treatment.transform([treat_norm])

    return np.hstack([X_comp, X_treat])

# -----------------
# 4. Image Blending with Padding Removal
# -----------------
def blend_images(images_to_blend):
    try:
        if not images_to_blend:
            return None, []

        ref_img = Image.open(os.path.join(IMAGE_DIR, images_to_blend[0]['image_name'])).convert('RGB')
        target_size = ref_img.size

        blended = np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)
        weight_map = np.zeros((target_size[1], target_size[0]), dtype=np.float32)

        total_weight = sum(p['weight'] for p in images_to_blend)
        used_images = []

        for p in images_to_blend:
            img = Image.open(os.path.join(IMAGE_DIR, p['image_name'])).convert('RGB').resize(target_size)
            img_array = np.array(img, dtype=np.float32)

            brightness = img_array.sum(axis=2)
            variation = img_array.max(axis=2) - img_array.min(axis=2)

            mask = ~((brightness < 20) & (variation < 5))
            mask_3d = mask[:, :, None]

            blended += img_array * p['weight'] * mask_3d
            weight_map += p['weight'] * mask

            used_images.append({
                "image_name": p['image_name'],
                "weight_percent": round((p['weight'] / total_weight) * 100, 2)
            })

        final = np.divide(
            blended,
            weight_map[:, :, None],
            out=np.zeros_like(blended),
            where=weight_map[:, :, None] != 0
        )

        final = np.clip(final, 0, 255).astype(np.uint8)

        rows = np.any(weight_map > 0, axis=1)
        cols = np.any(weight_map > 0, axis=0)

        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            final = final[rmin:rmax+1, cmin:cmax+1]

        Image.fromarray(final).save(BLENDED_IMAGE_PATH)
        return BLENDED_IMAGE_FILENAME, used_images

    except Exception as e:
        print("Blending error:", e)
        traceback.print_exc()
        return None, []

# -----------------
# 5. Four-Tier Strategy
# -----------------
def get_images_for_blending(comp, treat, mag):
    comp_n = normalize_string(comp)
    treat_n = normalize_string(treat)
    mag_n = normalize_string(mag)

    # Tier 1
    key1 = f"{comp_n} | {treat_n} | {mag_n}"
    if key1 in mag_class_image_list_map:
        imgs = mag_class_image_list_map[key1]
        w = 1 / len(imgs)
        return [{"image_name": i, "weight": w} for i in imgs], "Tier 1: Full Match"

    # Tier 2
    key2 = f"{comp_n} | {treat_n}"
    if key2 in class_image_list_map:
        imgs = class_image_list_map[key2]
        w = 1 / len(imgs)
        return [{"image_name": i, "weight": w} for i in imgs], "Tier 2: Class Match"

    # Tier 3
    if comp_n in comp_image_list_map:
        imgs = comp_image_list_map[comp_n]
        w = 1 / len(imgs)
        return [{"image_name": i, "weight": w} for i in imgs], "Tier 3: Composition Match"

    # Tier 4 (Model)
    X = preprocess_input(comp, treat)
    probs = model.predict_proba(X)[0]

    top_idx = np.argsort(probs)[::-1][:3]
    total = sum(probs[i] for i in top_idx if probs[i] >= 0.01)

    results = []
    for i in top_idx:
        if probs[i] < 0.01:
            continue
        img_name = image_map.get(model.classes_[i])
        if img_name:
            results.append({
                "image_name": img_name,
                "weight": probs[i] / total
            })

    return results, "Tier 4: Model-Based"

# -----------------
# 6. Routes
# -----------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs safely
        comp_input = request.form.get('composition', '').strip()
        treat_input = request.form.get('treatment', '').strip()
        mag_input = request.form.get('magnification', '').strip()  # Optional
        
        if not comp_input or not treat_input:
            return jsonify({'error': 'Please provide values for Composition and Treatment.'})
        
        # Get images to blend
        images_to_blend, strategy = get_images_for_blending(comp_input, treat_input, mag_input)
        
        if not images_to_blend:
            return jsonify({'error': 'No relevant images found or prediction confidence too low.'})
        
        # Blend images
        blended_filename, used_images_summary = blend_images(images_to_blend)
        
        if not blended_filename:
            return jsonify({'error': 'Failed to generate blended image.'})
        
        # ✅ Ensure all keys exist and are strings
        response = {
            'status': 'success',
            'composition_input': comp_input,
            'treatment_input': treat_input,
            'magnification_input': mag_input if mag_input else "",
            'image_src': blended_filename,
            'strategy': strategy,
            'used_images': used_images_summary
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Unexpected error: {str(e)}"})


# -----------------
# 7. Run App
# -----------------
if __name__ == "__main__":
    os.makedirs(IMAGE_DIR, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
