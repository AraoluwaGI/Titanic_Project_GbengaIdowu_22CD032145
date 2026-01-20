from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and encoders
MODEL_PATH = os.path.join('model', 'titanic_survival_model.pkl')
SEX_ENCODER_PATH = os.path.join('model', 'sex_encoder.pkl')
EMBARKED_ENCODER_PATH = os.path.join('model', 'embarked_encoder.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    sex_encoder = joblib.load(SEX_ENCODER_PATH)
    embarked_encoder = joblib.load(EMBARKED_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✓ Model and encoders loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    sex_encoder = None
    embarked_encoder = None
    scaler = None

@app.route('/')
def home():
    """Render the home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or sex_encoder is None or embarked_encoder is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded properly. Please check server logs.'
            }), 500

        # Get input data from the form
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']

        # Validate inputs
        if pclass not in [1, 2, 3]:
            return jsonify({'error': 'Passenger Class must be 1, 2, or 3'}), 400
        
        if age < 0 or age > 100:
            return jsonify({'error': 'Age must be between 0 and 100'}), 400
        
        if fare < 0:
            return jsonify({'error': 'Fare cannot be negative'}), 400

        # Encode categorical variables
        try:
            sex_encoded = sex_encoder.transform([sex])[0]
            embarked_encoded = embarked_encoder.transform([embarked])[0]
        except ValueError as e:
            return jsonify({'error': f'Invalid input: {str(e)}'}), 400

        # Prepare features in the correct order
        # Order: Pclass, Sex_Encoded, Age, Fare, Embarked_Encoded
        features = pd.DataFrame({
            'Pclass': [pclass],
            'Sex_Encoded': [sex_encoded],
            'Age': [age],
            'Fare': [fare],
            'Embarked_Encoded': [embarked_encoded]
        })

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        # Prepare response
        result = {
            'prediction': int(prediction),
            'prediction_text': 'Survived' if prediction == 1 else 'Did Not Survive',
            'survival_probability': float(probability[1] * 100),
            'death_probability': float(probability[0] * 100),
            'passenger_info': {
                'class': pclass,
                'sex': sex,
                'age': age,
                'fare': f'£{fare:.2f}',
                'embarked': embarked
            }
        }

        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoders_loaded': sex_encoder is not None and embarked_encoder is not None,
        'scaler_loaded': scaler is not None
    })

if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)