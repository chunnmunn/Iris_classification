from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load the model and scaler
model = joblib.load('saved_model1.pkl')
scaler = joblib.load('scaler.save')

app = Flask(__name__)

# Configure the image upload folder
IMG_FOLDER = os.path.join('static', 'IMG')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form data
        sepal_length = request.form['SepalLength']
        sepal_width = request.form['SepalWidth']
        petal_length = request.form['PetalLength']
        petal_width = request.form['PetalWidth']
        
        # Prepare data for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=float)
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        predicted_class = prediction[0]
        
        # Map prediction to image
        image_filename = f'{predicted_class}.png'
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        
        return render_template('index.html', prediction=predicted_class, image=image_path)
    
    # Render the home page for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
