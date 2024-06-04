# Dog Classifier

This project is a simple web application that uses a CNN to classify images of my dogs Charlie and Rosie. Other than making my wife and I laugh it is utterly useless but at leat showcases my ability to build some simple stuff. 

The model uses a pre-trained InceptionV3 architecture, which has been fine-tuned with over 280 images for this specific task of distinguishing between images of my two dogs, Rosie and Charlie. 
Data augmentation techniques, such as rotation, width and height shifts, shear, zoom, and horizontal flip, were employed to enhance the robustness of the model during training. The model achieved a test accuracy of over 95% on unseen data,

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/dog-classifier.git
   cd dog-classifier
Set up a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

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

