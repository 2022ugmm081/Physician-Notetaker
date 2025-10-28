# Physician Notetaker 
Overview 
This project builds an NLP pipeline that extracts structured medical details from a doctor-
patient conversation transcript. 
It performs Named Entity Recognition (NER), Summarization, Keyword Extraction, and 
Emotion/Intent Classification to generate a clean medical report in JSON format. 
 
Tech Stack 
• Python 
• Transformers (Hugging Face) 
• BART / KeyBERT 
• Regex 
• Flask (optional for deployment) 
 
Task 1: Medical NLP Summarization 
Goal: Extract and summarize clinical information from the transcript. 
Pipeline Steps 
1. NER Extraction — Identify medical entities (symptoms, treatment, diagnosis, 
prognosis) using d4data/biomedical-ner-all 
2. Summarization — Generate a short report using facebook/bart-large-cnn 
3. Keyword Extraction — Extract key medical phrases using KeyBERT 
4. JSON Output — Create a structured report with: 
JSON 
{ 
  "Patient_Name": "Ms. Jones", 
  "Symptoms": ["Neck pain", "Back pain", "Head impact"], 
  "Diagnosis": "Whiplash injury", 
  "Treatment": ["Painkillers", "Physiotherapy"], 
  "Current_Status": "Occasional backache", 
  "Prognosis": "Full recovery within six months" 
} 
 
Task 1 - Abstract 
This project demonstrates a robust Hybrid NLP Pipeline designed to solve a real-world 
business problem: converting unstructured medical conversations into accurate, structured 
data. 
The core of my approach was to reject a single-model solution and instead build a multi-
stage pipeline where specialized models and custom logic work together. 
My pipeline executes in four stages: 
1. High-Level Summarization: A BART model generates a narrative summary of the 
consultation. 
2. Specialized NER: A pre-trained d4data/biomedical-ner-all model extracts specific 
medical entities (symptoms, treatments, etc.). 
3. Intelligent Post-Processing: This is the key innovation. I developed custom logic to 
refine the model's raw output. This logic contextually combines related entities (e.g., 
Biological_structure "neck" + Sign_symptom "pain" = "neck pain") and filters out 
negated symptoms (e.g., "no fever"). 
4. Data Assembly: The clean data from all models is assembled into a final JSON object 
and a human-readable report. 
This hybrid method proves to be far more accurate and robust than a single model, 
demonstrating an ability to build practical, real-world AI solutions that go beyond simple 
model deployment. 
 
 
Task 2: Patient Emotion & Intent Detection 
Goal: Identify the sentiment and intent of the patient during conversation. 
Pipeline Steps 
1. Extract all patient responses using regex 
2. Use Zero-Shot Classification (facebook/bart-large-mnli) 
3. Predict both: 
JSON 
{ 
  "Sentiment": "Reassured", 
  "Intent": "Reporting symptoms" 
} 
 
