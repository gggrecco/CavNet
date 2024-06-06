# Dog Classifier

This project is a simple web application that uses a CNN to classify images of my dogs Charlie and Rosie. 
Initially I thought I would build a large more complex model to identify different breeds of dogs but the compute and images to do it was a bit too much for a random side project. 
Other than making my wife and I laugh it is utterly useless but at leat showcases my ability to build some simple stuff. 

The model uses a pre-trained InceptionV3 architecture, which has been fine-tuned with over 280 images for this specific task of distinguishing between images of my two dogs, Rosie and Charlie. 
Data augmentation techniques, such as rotation, width and height shifts, shear, zoom, and horizontal flip, were employed to enhance the robustness of the model during training. The model achieved a test accuracy of over 95% on unseen data. here are some metrics regarding training/validation loss and accuracy as well as test performance: <img width="665" alt="image" src="https://github.com/gggrecco/CavNet-CNN-for-Dog-Classification-/assets/72873244/11e490f9-00fc-4f9a-9222-bc63dfd166bc">

Here's a sample output from a test image: <img width="958" alt="image" src="https://github.com/gggrecco/CavNet-CNN-for-Dog-Classification-/assets/72873244/78353f21-19d9-431f-a7c0-75c5d96d9c7e">

## Setup
clone the repository

Install dependencies:
pip install -r requirements.txt

Run the Flask app:
python app.py

Open index.html in your browser

Project Structure
app.py: The Flask backend server that handles image uploads and makes predictions.
requirements.txt: The Python dependencies required for the project.
index.html: The frontend HTML file.
script.js: The JavaScript file to handle image uploads and display results.
saved_model_dir/: The directory containing the saved model 

