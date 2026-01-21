# Titanic Survival Prediction System

A machine learning web application that predicts passenger survival on the Titanic using a Random Forest Classifier.

## 🎯 Project Overview

- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~81%
- **Features Used**: Pclass, Sex, Age, Fare, Embarked
- **Framework**: Flask
- **Deployment**: Render.com

## 📊 Model Development

### Dataset
- **Source**: Kaggle - Titanic: Machine Learning from Disaster
- **Total Passengers**: 891
- **Survival Rate**: 38.38%

### Preprocessing Steps
1. **Missing Value Handling**:
   - Age: Filled with median (28 years)
   - Fare: Filled with median
   - Embarked: Filled with mode (Southampton)

2. **Feature Selection**:
   - Selected 5 features from 7 available
   - Features: Pclass, Sex, Age, Fare, Embarked

3. **Encoding**:
   - Sex: LabelEncoder (male=1, female=0)
   - Embarked: LabelEncoder (C=0, Q=1, S=2)

4. **Feature Scaling**:
   - StandardScaler applied to all features
   - Mean ≈ 0, Std Dev ≈ 1

### Model Training
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 5
  - min_samples_leaf: 2
  - random_state: 42

### Model Evaluation
```
Accuracy: 81.56%

Classification Report:
                   precision    recall  f1-score
Did Not Survive       0.83      0.87      0.85
Survived              0.78      0.71      0.74

Overall accuracy: 0.82
```

### Feature Importance
1. Sex: 0.32 (Most important)
2. Fare: 0.26
3. Age: 0.23
4. Pclass: 0.15
5. Embarked: 0.04 (Least important)

## 🚀 Local Setup

### Prerequisites
- Python 3.11+
- pip

### Installation
```bash
# Clone repository
git clone https://github.com/YourUsername/Titanic_Project_GbengaIdowu_22CD032145.git
cd Titanic_Project_GbengaIdowu_22CD032145

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### Usage
1. Open browser: `http://127.0.0.1:5000`
2. Enter passenger details
3. Click "Predict Survival"
4. View prediction and probability

## 🌐 Live Demo

**URL**: https://titanic-survival-predictor-gbengaidowu.onrender.com

**Note**: First load may take 30-60 seconds (free tier)

## 📁 Project Structure
```
Titanic_Project_GbengaIdowu_22CD032145/
├── app.py                          # Flask application
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
├── Titanic_hosted_webGUI_link.txt  # Deployment info
├── model/
│   ├── model_building.ipynb        # Training notebook
│   ├── titanic_survival_model.pkl  # Trained model
│   ├── sex_encoder.pkl             # Sex encoder
│   ├── embarked_encoder.pkl        # Embarked encoder
│   └── scaler.pkl                  # Feature scaler
├── static/
│   └── style.css                   # Styling
└── templates/
    └── index.html                  # Frontend
```

## 🔧 API Endpoints

### `GET /`
Renders the main web interface

### `POST /predict`
Predicts survival for given passenger data

**Request Body** (form-data):
```
pclass: int (1, 2, or 3)
sex: string ("male" or "female")
age: float (0-100)
fare: float (≥0)
embarked: string ("C", "Q", or "S")
```

**Response**:
```json
{
  "prediction": 1,
  "prediction_text": "Survived",
  "survival_probability": 85.3,
  "death_probability": 14.7,
  "passenger_info": {...}
}
```

### `GET /health`
Health check endpoint

## 🛠 Technologies Used

- **Backend**: Flask 3.0.0
- **ML Library**: scikit-learn 1.3.2
- **Data Processing**: pandas 2.1.4, numpy 1.26.2
- **Model Persistence**: joblib 1.3.2
- **Deployment**: Render.com with Gunicorn

## 👨‍💻 Author

**Gbenga Idowu**  
Matric No: 22CD032145

## 📄 License

This project is for educational purposes.
