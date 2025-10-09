from flask import Flask, render_template, request, jsonify
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import base64
import os

app = Flask(__name__)

# Initialize Vertex AI
PROJECT_ID = "loyal-world-472411-s7"
LOCATION = "us-central1"

# Use key.json for local dev, automatic auth for Cloud Run
if os.path.exists("key.json"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.0-flash-001")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        image_file = request.files['image']
        prompt = request.form['prompt']
        
        # Read image
        image_bytes = image_file.read()
        image_part = Part.from_data(image_bytes, mime_type=image_file.content_type)
        
        # Generate response
        response = model.generate_content([prompt, image_part])
        
        return jsonify({'analysis': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)