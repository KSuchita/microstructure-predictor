from flask import Flask, render_template, request, jsonify
from joblib import load
from PIL import Image
import pandas as pd
import numpy as np
import os
import shutil

app = Flask(__name__)

# Define constants
IMAGE_DIR = 'static/Resized'
OUTPUT_DIR = 'static'
BLENDED_IMAGE_FILENAME = 'blended_output.png'
BLENDED_IMAGE_PATH = os.path.join(OUTPUT_DIR, BLENDED_IMAGE_FILENAME)

# --- 1. Load Model and Preprocessing Components ---
try:
    model = load('model.joblib')
    mlb_composition = load('mlb_composition.joblib')
    lb_treatment = load('lb_treatment.joblib')
    image_map = load('image_map.joblib')
    
    print("Model and components loaded successfully.")

except Exception as e:
    print(f"Error loading components: {e}")
    # exit() 

# --- 2. Preprocessing Function for New Input (Unchanged) ---
def preprocess_input(comp_input, treat_input):
    """Preprocesses user input into the model's feature vector (X)."""
    comp_list = [comp_input.split()]
    comp_ohe = mlb_composition.transform(comp_list)
    treat_array = np.array([treat_input])
    treat_ohe = lb_treatment.transform(treat_array)
    X_pred = np.hstack([comp_ohe, treat_ohe])
    return X_pred

# --- 3. Image Blending Function (NEW) ---
def blend_images(predictions):
    """
    Blends the top N images based on their confidence scores using NumPy and PIL.
    Saves the single blended image to static/blended_output.png.
    """
    if not predictions:
        return None

    # Use the first image's size as the reference size for blending
    first_img_path = os.path.join(IMAGE_DIR, predictions[0]['image_name'])
    try:
        ref_img = Image.open(first_img_path).convert('RGB')
        target_size = ref_img.size
        blended_array = np.zeros((*target_size[::-1], 3), dtype=np.float32)
        
        # Calculate the sum of probabilities for normalization
        total_prob = sum(p['probability'] for p in predictions)

        for p in predictions:
            img_path = os.path.join(IMAGE_DIR, p['image_name'])
            
            # Use the actual probability, normalized by the total sum
            weight = p['probability'] / total_prob 
            
            # Open, resize to target size, and convert to NumPy array
            img = Image.open(img_path).convert('RGB').resize(target_size)
            img_array = np.array(img, dtype=np.float32)
            
            # Accumulate the weighted pixel array
            blended_array += (img_array * weight)
            
        # Ensure pixel values are clipped to 0-255 range and converted to UINT8
        blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
        
        # Convert NumPy array back to PIL Image and save
        blended_img = Image.fromarray(blended_array)
        blended_img.save(BLENDED_IMAGE_PATH)
        
        return BLENDED_IMAGE_FILENAME

    except Exception as e:
        print(f"Error during image blending: {e}")
        # Copy a simple 'not found' image or handle gracefully on the frontend
        return None

# --- 4. Flask Routes ---

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
        top_n_indices = np.argsort(probabilities)[::-1][:3] 
        
        predictions = []
        for index in top_n_indices:
            microstructure_id = model.classes_[index] 
            probability = probabilities[index]
            
            if probability < 0.01: continue 

            predicted_image_name = image_map.get(microstructure_id, 'image_not_found.png')
            
            predictions.append({
                'probability': float(f'{probability:.3f}'),
                'image_name': predicted_image_name # Note: We use image_name here for the blend function
            })
        
        if not predictions:
            return jsonify({'error': 'Prediction confidence too low for the given input.'})
            
        # *** Call the blending function ***
        blended_filename = blend_images(predictions)
        
        if blended_filename:
            response = {
                'status': 'success',
                'composition_input': comp_input,
                'treatment_input': treat_input,
                # Return only the path to the single blended image
                'image_src': blended_filename 
            }
        else:
            response = {'error': 'Failed to generate blended image.'}
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f"An error occurred during prediction: {e}"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)