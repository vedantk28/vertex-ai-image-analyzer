from flask import Flask, render_template, request, jsonify
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import os
from datetime import datetime
import json

app = Flask(__name__)

PROJECT_ID = "loyal-world-472411-s7"
LOCATION = "us-central1"

if os.path.exists("key.json"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.0-flash-001")

# Enhanced prompts for agriculture & poultry
QUICK_PROMPTS = {
    "plant_disease": "Analyze this plant image in detail. Identify: 1) Plant species 2) Any diseases/pests present 3) Disease severity (mild/moderate/severe) 4) Nutritional deficiencies 5) Recommended treatments 6) Preventive measures. Format as structured analysis.",
    "soil_analysis": "Examine this soil image. Provide: 1) Soil type and texture 2) Moisture level assessment 3) Visible nutrient indicators 4) Soil health score 5) Recommended amendments 6) Best crops for this soil. Be specific and actionable.",
    "livestock_health": "Assess this livestock image thoroughly. Report: 1) Animal species and breed 2) Physical condition score 3) Signs of disease/distress 4) Behavioral indicators 5) Recommended actions 6) Veterinary consultation needed? Provide confidence level.",
    "poultry_diagnosis": "Analyze this poultry image comprehensively. Check: 1) Bird species 2) Physical appearance and posture 3) Feather/comb/eye condition 4) Possible diseases 5) Fecal quality if visible 6) Treatment recommendations 7) Biosecurity measures needed.",
    "equipment_check": "Evaluate this farm equipment/machinery. Assess: 1) Equipment type and model 2) Visible condition and wear 3) Maintenance issues 4) Safety concerns 5) Performance optimization tips 6) Estimated remaining lifespan."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-prompts', methods=['GET'])
def get_prompts():
    return jsonify(QUICK_PROMPTS)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        image_file = request.files['image']
        prompt = request.form['prompt']
        category = request.form.get('category', 'general')
        
        image_bytes = image_file.read()
        image_part = Part.from_data(image_bytes, mime_type=image_file.content_type)
        
        # Enhanced prompt with structure
        enhanced_prompt = f"""{prompt}

IMPORTANT: Structure your response with clear sections using these headers:
## Overview
## Key Findings  
## Severity/Condition Score
## Detailed Analysis
## Recommendations
## Preventive Measures

Use bullet points and be specific with actionable advice."""

        response = model.generate_content([enhanced_prompt, image_part])
        
        analysis_result = {
            'analysis': response.text,
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'image_name': image_file.filename
        }
        
        return jsonify(analysis_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    try:
        images = request.files.getlist('images')
        prompt = request.form['prompt']
        results = []
        
        for img in images[:5]:  # Limit to 5 images
            image_bytes = img.read()
            image_part = Part.from_data(image_bytes, mime_type=img.content_type)
            response = model.generate_content([prompt, image_part])
            results.append({
                'filename': img.filename,
                'analysis': response.text
            })
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)