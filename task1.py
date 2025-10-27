# -*- coding: utf-8 -*-
import re
import json
import datetime as dt
import textwrap
import numpy as np
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from keybert import KeyBERT

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Helper Function to Fix JSON Serialization ---
def convert_to_serializable(obj):
    """Recursively converts numpy types to standard Python types."""
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

# --- Transcript Analysis Code (from Task 1) ---

SAMPLE_TRANSCRIPT = " "
"""
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

def load_models():
    """Loads all the required AI models for Task 1."""
    print("Loading summarization model...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    print("Loading KeyBERT model...")
    kw_model = KeyBERT()

    print("Loading NER model...")
    ner_model_name = "d4data/biomedical-ner-all"
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
    ner_pipeline = pipeline('ner', model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")
    
    return summarizer, kw_model, ner_pipeline

def merge_entities(entities, text):
    """Merges adjacent entities of the same type."""
    if not entities:
        return []

    entities = sorted(entities, key=lambda i: i['start'])
    merged_entities = []
    current_entity = entities[0].copy()

    for next_ent in entities[1:]:
        if (current_entity['entity_group'] == next_ent['entity_group'] and
            current_entity['end'] >= next_ent['start'] - 1):
            # Merge entities
            current_entity['end'] = max(current_entity['end'], next_ent['end'])
            current_entity['word'] = text[current_entity['start']:current_entity['end']].strip().replace("##", "")
            current_entity['score'] = (current_entity['score'] + next_ent['score']) / 2
        else:
            # Add current entity and start a new one
            merged_entities.append(current_entity)
            current_entity = next_ent.copy()

    merged_entities.append(current_entity)
    return merged_entities

def combine_body_symptoms(ner_results):
    """Combines 'Biological_structure' + 'Sign_symptom' into a single symptom."""
    combined = []
    i = 0
    while i < len(ner_results):
        current = ner_results[i]

        if (i + 1 < len(ner_results) and
            current['entity_group'] == 'Biological_structure' and
            ner_results[i+1]['entity_group'] == 'Sign_symptom' and
            current['end'] >= ner_results[i+1]['start'] - 5): # 5 char gap tolerance

            combined_ent = current.copy()
            combined_ent['word'] = current['word'] + ' ' + ner_results[i+1]['word']
            combined_ent['entity_group'] = 'Sign_symptom'
            combined_ent['end'] = ner_results[i+1]['end']
            combined_ent['score'] = (current['score'] + ner_results[i+1]['score']) / 2
            combined.append(combined_ent)
            i += 2 # Skip next entity
        else:
            combined.append(current)
            i += 1
    return combined

def analyze_transcript(transcript, summarizer, kw_model, ner_pipeline):
    """Runs the full analysis pipeline on the transcript."""
    
    # Run NER
    ner_results = ner_pipeline(transcript)
    ner_results = merge_entities(ner_results, transcript)
    ner_results = combine_body_symptoms(ner_results)

    # Extract structured info
    name_match = re.search(r"Good morning,\s*(Ms\.|Mr\.|Mrs\.|Dr\.)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", transcript)
    patient_name = name_match.group(1) + " " + name_match.group(2) if name_match else "Not mentioned"

    symptoms = []
    treatments = []
    diagnoses = []

    for ent in ner_results:
        group = ent.get('entity_group', '')
        word = ent['word'].strip()
        if not word:
            continue

        if group == 'Sign_symptom':
            # Check context for negation
            context = transcript[max(0, ent['start'] - 50): ent['end'] + 50].lower()
            if not re.search(r'\b(no|not|don\'t|didn\'t|no longer|none|denies)\b', context):
                symptoms.append(word)

        elif group == 'Therapeutic_procedure':
            treatments.append(word)

    # Manual extraction for items models might miss
    if not diagnoses:
        diag_match = re.search(r"(whiplash injury)", transcript, re.IGNORECASE)
        if diag_match:
            diagnoses.append(diag_match.group(1))

    treat_match = re.search(r"(painkillers)", transcript, re.IGNORECASE)
    if treat_match and "painkillers" not in treatments:
        treatments.append(treat_match.group(1))

    # Clean up and format
    symptoms = list(set(symptoms)) if symptoms else ["Not mentioned"]
    treatments = list(set(treatments)) if treatments else ["Not mentioned"]
    diagnosis = ", ".join(set(diagnoses)) if diagnoses else "Not mentioned"

    # Run other models
    summary = summarizer(transcript, max_length=130, min_length=60, do_sample=False)[0]['summary_text']
    keywords = [kw[0] for kw in kw_model.extract_keywords(transcript, top_n=8, stop_words='english')]

    # Extract prognosis and status
    prognosis_match = re.search(r"full recovery within (.*?)\.", transcript, re.IGNORECASE)
    prognosis = prognosis_match.group(0) if prognosis_match else "Not mentioned"

    status_match = re.search(r"occasional (.*?)\.", transcript, re.IGNORECASE)
    current_status = status_match.group(0) if status_match else "Back to normal"

    # Compile final reports
    final_report = {
        "Patient_Name": patient_name,
        "Symptoms": symptoms,
        "Diagnosis": diagnosis,
        "Treatment": treatments,
        "Current_Status": current_status,
        "Prognosis": prognosis,
    }

    return final_report, summary, keywords, ner_results

def generate_medical_report(final_report, summary):
    """Formats the analysis into a text-based medical report."""
    hospital_name = "[Hospital Name/Clinic Name]"
    physician_name = "[Physician Name]"

    symptoms_str = "- " + "\n- ".join(final_report['Symptoms']) if final_report['Symptoms'] != ["Not mentioned"] else "Not mentioned"
    treatments_str = "- " + "\n- ".join(final_report['Treatment']) if final_report['Treatment'] != ["Not mentioned"] else "Not mentioned"

    wrapped_summary = textwrap.fill(summary, width=70)

    medical_report = f"""
==============================================
         MEDICAL CONSULTATION REPORT
==============================================

Hospital/Clinic: {hospital_name}
Date: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Physician: {physician_name}

-------
PATIENT INFORMATION
-------
Patient Name: {final_report['Patient_Name']}

-------
SUMMARY OF CONSULTATION
-------
{wrapped_summary}

-------
CLINICAL FINDINGS
-------
Diagnosis:
{final_report['Diagnosis']}

Symptoms Reported:
{symptoms_str}

Current Status:
{final_report['Current_Status']}

-------
TREATMENT & PROGNOSIS
-------
Treatment Provided/Recommended:
{treatments_str}

Prognosis:
{final_report['Prognosis']}

==============================================
                END OF REPORT
==============================================
"""
    return medical_report.strip()


# --- Load Models Globally ---
print("Loading models for Task 1. This may take a minute...")
try:
    g_summarizer, g_kw_model, g_ner_pipeline = load_models()
    print("Task 1 models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    g_summarizer, g_kw_model, g_ner_pipeline = (None, None, None)


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('task1_index.html', sample_transcript=SAMPLE_TRANSCRIPT)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles the analysis request."""
    if not g_summarizer:
        return jsonify({"error": "Models are not loaded."}), 500

    data = request.json
    transcript = data.get('transcript', SAMPLE_TRANSCRIPT)

    try:
        # Run the analysis
        final_report, summary, keywords, ner_results = analyze_transcript(
            transcript, g_summarizer, g_kw_model, g_ner_pipeline
        )

        # Generate the text report
        full_report_text = generate_medical_report(final_report, summary)

        # --- FIX: Convert numpy types before sending ---
        serializable_ner_results = convert_to_serializable(ner_results)

        # Return all results as JSON
        return jsonify({
            "full_report_text": full_report_text,
            "structured_report": final_report,
            "summary": summary,
            "keywords": keywords,
            "ner_debug": serializable_ner_results # <-- Use the converted results
        })

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": str(e)}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Running on port 5001
    app.run(debug=True, host='0.0.0.0', port=5001)

