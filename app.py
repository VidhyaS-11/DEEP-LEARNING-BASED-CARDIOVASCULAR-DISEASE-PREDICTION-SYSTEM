from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("model_conv1d.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from form
        input_features = [
            float(request.form['age']),
            float(request.form['height']),
            float(request.form['weight']),
            float(request.form['ap_hi']),
            float(request.form['ap_lo']),
            float(request.form['cholesterol']),
            float(request.form['gluc']),
            float(request.form['smoke']),
            float(request.form['alco']),
            float(request.form['active']),
            float(request.form['gender'])
        ]

        # Calculate BMI
        bmi = input_features[2] / ((input_features[1] / 100) ** 2)
        input_features.append(bmi)

        # Convert to numpy array
        input_data = np.array(input_features).reshape(1, -1)

        # Scale the data
        input_scaled = scaler.transform(input_data)

        # Reshape for Conv1D: (1, 12, 1)
        input_reshaped = input_scaled.reshape(1, 12, 1)

        # Predict
        prediction = model.predict(input_reshaped)
        result = "😨 High Risk of Heart Disease" if prediction[0][0] > 0.5 else " 😊 Low Risk of Heart Disease"

        return render_template('result.html', prediction=result)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
