import os
import cv2
import numpy as np
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg', 'webp'}
MODEL_MAPPING = {
    'AlexNet': 'AlexNetModel.h5',
    'LeNet': 'LeNetModel.h5',
    'ResNet': 'ResNetModel.h5'
}

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(227, 227)):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image at path {img_path}")
        return None
    img = cv2.resize(img, target_size)  # Resize image based on target size
    img = img.astype("float32") / 255.0
    return img


def cal_cloud_percent_and_label(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(image, 140, 1, cv2.THRESH_BINARY)
    h, w = image.shape[:2]
    total = sum(map(sum, thresh))
    percent = total / (h * w) * 100
    percent = float(percent)  # Convert percent to float explicitly
    if 100 >= percent > 90:
        cloudy_label = "Cloudy"
    elif 90 >= percent > 70:
        cloudy_label = "Mostly Cloudy"
    elif 70 >= percent > 30:
        cloudy_label = "Partly Cloudy"
    else:
        cloudy_label = "Clear"
    return percent, str(cloudy_label)  # Convert cloudy_label to string




def predict_weather(image_path, cloud_percentage, model):
    if model == 'LeNet':
        img = preprocess_image(image_path, target_size=(32, 32))
    else:
        img = preprocess_image(image_path, target_size=(227, 227))

    if img is None:
        return "Error: Unable to preprocess image."
    
    img = np.expand_dims(img, axis=0)
    cloud_percentage = np.array([[cloud_percentage]])  # Reshape cloud percentage for model input
    model_filename = MODEL_MAPPING.get(model)  # Access model filename using model_name
    if model_filename is None:
        return f"Error: Model '{model}' not found in MODEL_MAPPING."

    model = load_model(model_filename)
    prediction = model.predict([img, cloud_percentage])
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_weather', methods=["POST"])
def predict_weather_route():
    if request.method == "POST":
        model_name = request.form.get("model")
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Calculate cloud percentage and label
            percent, _ = cal_cloud_percent_and_label(image_path)
            
            # Load model
            model_filename = MODEL_MAPPING[model_name]
            model = load_model(model_filename)
            
            # Perform prediction
            prediction = predict_weather(image_path, percent, model_name)
            
            # Pass variables to the result.html template
            return render_template("result.html", 
                                    image_filename=filename,  # Pass the image filename
                                    cloud_percent=percent, 
                                    prediction=prediction)
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)
