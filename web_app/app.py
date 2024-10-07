from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
from models.architectures import create_model  # Import your model

app = Flask(__name__)

# Load your model
model = create_model(...)  # Initialize your model
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    # Preprocess image
    # Make prediction
    # Return result
    return jsonify({'prediction': 'Mild Cognitive Impairment', 'confidence': '85%'})

if __name__ == '__main__':
    app.run(debug=True)