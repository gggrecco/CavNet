import zipfile
import os
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Unzip the model
def unzip_model():
    with zipfile.ZipFile("saved_model_dir.zip", "r") as zip_ref:
        zip_ref.extractall(".")

# Ensure the model is unzipped
if not os.path.exists("saved_model_dir"):
    unzip_model()

# Load the model
model = tf.saved_model.load("saved_model_dir")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model(img_array, training=False)
    result = 'Rosie' if prediction > 0.5 else 'Charlie'

    os.remove(img_path)
    return jsonify({'result': result})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000)
