# Physician Notetaker
## Overview

Physician Notetaker is a Natural Language Processing (NLP) pipeline designed to extract structured clinical information from doctor–patient conversation transcripts.
It performs:

Named Entity Recognition (NER)

Summarization

Keyword Extraction

Emotion & Intent Classification

The final output is a clean and structured medical report in JSON format.

Tech Stack

Language: Python

Libraries & Models:

Transformers (Hugging Face)

spaCy / BART / KeyBERT

Regex

Deployment : Flask

### Task 1: Medical NLP Summarization
Goal:
Extract and summarize clinical details from a doctor–patient conversation.

Pipeline Steps:
NER Extraction: Identify medical entities such as symptoms, treatments, diagnoses, prognosis using 

d4data/biomedical-ner-all

Summarization: Generate a concise medical report using

facebook/bart-large-cnn

Keyword Extraction: Extract key medical phrases using
KeyBERT


JSON Output: Produce structured data such as:

{
  "Patient_Name": "Ms. Jones",
  "Symptoms": ["Neck pain", "Back pain", "Head impact"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["Painkillers", "Physiotherapy"],
  "Current_Status": "Occasional backache",
  "Prognosis": "Full recovery within six months"
}

### Task 2: Patient Emotion & Intent Detection

Goal:
Identify the sentiment and intent behind the patient's responses.

Pipeline Steps:
Extract all patient responses using Regex

Apply Zero-Shot Classification with
facebook/bart-large-mnli

Generate both Sentiment and Intent outputs as JSON:

{
  "Sentiment": "Reassured",
  "Intent": "Reporting symptoms"
}

### Folder Structure
Emitrr_Physician_Notetaker_VishalMaurya/
│
├── README.md
├── requirements1.txt
├── requirements2.txt
├── Emitrr_Physician_Notetaker_Task_1.ipynb
├── Emitrr_Physician_Notetaker_Task_2.ipynb

#### How to Run
Task 1: Medical NLP Summarization

pip install -r requirements1.txt

python task_1.py

Task 2: Emotion & Intent Detection

pip install -r requirements2.txt

python task_2.py

Requirements (Summary)
flask
transformers
torch
sentencepiece
accelerate
keybert
numpy

Output
Task 1:

Structured JSON file

📄 Formatted medical summary report

Task 2:

JSON output with Sentiment & Intent

Author

Vishal Maurya
vishal.gusknp2022@gmail.com

National Institute of Technology, Jamshedpur
