# -*- coding: utf-8 -*-
import re
import json
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Transcript Analysis Code (from Task 2) ---

SAMPLE_TRANSCRIPT = """
Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: Good morning, doctor. I’m doing better, but I still have some discomfort now and then.
Physician: I understand you were in a car accident last September. Can you walk me through what happened?
Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
Physician: That sounds like a strong impact. Were you wearing your seatbelt?
Patient: Yes, I always do.
Physician: What did you feel immediately after the accident?
Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
Physician: Did you seek medical attention at that time?
Patient: Yes, I went to Moss Bank Accident and Emergency. They said it was a whiplash injury. They gave me advice and sent me home.
Physician: How did things progress after that?
Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers. It started improving after that, but I had ten physiotherapy sessions to help with the stiffness.
Physician: Are you still experiencing pain now?
Patient: It’s not constant, but I do get occasional backaches.
Physician: Any emotional effects?
Patient: No, I don’t feel nervous driving or have emotional issues.
Physician: Has this impacted your daily life or work?
Patient: I took a week off work, but then returned to normal.
Physician: Everything looks good physically. No lasting damage.
Patient: That’s a relief.
Physician: You’ll make a full recovery within six months. No long-term impact expected.
Patient: That’s great to hear.
"""

def load_model():
    """Loads the zero-shot classification model."""
    print("Loading zero-shot-classification model...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("Model loaded successfully.")
    return classifier

def extract_patient_dialogs(transcript):
    """
    Uses regex to find all lines spoken by the "Patient"
    and joins them into a single string.
    """
    patient_lines = re.findall(r"Patient: (.*?)(?=\nPhysician:|\nPatient:|$)", transcript, re.DOTALL)
    patient_text = ' '.join([line.strip().replace('\n', ' ') for line in patient_lines])
    return patient_text if patient_text else "No patient dialogue found."

def analyze_sentiment_intent(patient_text, classifier):
    """Runs the sentiment and intent analysis."""
    sentiment_labels = ["Anxious", "Neutral", "Reassured"]
    intent_labels = ["Seeking reassurance", "Reporting symptoms", "Expressing concern"]

    sentiment_result = classifier(patient_text, sentiment_labels)
    intent_result = classifier(patient_text, intent_labels)

    top_sentiment = sentiment_result['labels'][0]
    top_intent = intent_result['labels'][0]

    final_json_output = {
        "Sentiment": top_sentiment,
        "Intent": top_intent
    }
    return final_json_output

# --- Load Model Globally ---
print("Loading model for Task 2...")
try:
    g_classifier = load_model()
except Exception as e:
    print(f"Error loading model: {e}")
    g_classifier = None

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('task2_index.html', sample_transcript=SAMPLE_TRANSCRIPT)

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    """Handles the sentiment analysis request."""
    if not g_classifier:
        return jsonify({"error": "Model is not loaded."}), 500

    data = request.json
    transcript = data.get('transcript', SAMPLE_TRANSCRIPT)
    
    try:
        patient_text = extract_patient_dialogs(transcript)
        if patient_text == "No patient dialogue found.":
            return jsonify({"error": "No patient dialogue found in the transcript."}), 400

        result = analyze_sentiment_intent(patient_text, g_classifier)
        
        return jsonify(result)

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": str(e)}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Running on port 5002
    app.run(debug=True, host='0.0.0.0', port=5002)
