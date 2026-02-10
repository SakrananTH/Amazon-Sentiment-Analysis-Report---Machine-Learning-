import os
import joblib
import time
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pythainlp # Import needed for custom tokenizer

app = Flask(__name__)
CORS(app)

# --- Configuration ---
MODEL_PATH = 'sentiment_model.pkl'
CSV_PATH = '9.synthetic_amazon_like_reviews_subset_5000.csv'

# --- Custom Tokenizer (Required for joblib to load the model) ---
def custom_tokenizer(text):
    return pythainlp.word_tokenize(text, engine='newmm', keep_whitespace=False)
# ---------------------------------------------------------------

# --- Load Model ---
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    start_time = time.time()
    try:
        prediction = model.predict([text])[0]
        confidence = float(model.predict_proba([text]).max())
        latency = (time.time() - start_time) * 1000
        
        return jsonify({
            'sentiment': prediction,
            'confidence': confidence,
            'latency': latency
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data', methods=['GET'])
def get_data():
    if not os.path.exists(CSV_PATH):
        return jsonify({'error': 'CSV file not found'}), 404
    
    try:
        # Read only first 10 rows and specific columns to save bandwidth
        df = pd.read_csv(CSV_PATH, usecols=['text', 'label']).head(10)
        data = df.to_dict(orient='records')
        total_rows = 5000 # Hardcoded or fully read length, hardcoded for speed as per file name
        return jsonify({
            'data': data,
            'total_count': total_rows
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
