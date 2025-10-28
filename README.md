# Physician Notetaker 
### Overview 
This project builds an NLP pipeline that extracts structured medical details from a doctor-
patient conversation transcript. 
It performs Named Entity Recognition (NER), Summarization, Keyword Extraction, and 
Emotion/Intent Classification to generate a clean medical report in JSON format. 
 

### Tech Stack 
• Python 

• Transformers (Hugging Face) 

• BART / KeyBERT 

• Regex 

• Flask (optional for deployment) 
 
### Task 1: Medical NLP Summarization 
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
 
### Task 1 - Summary 
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
 
 
### Task 2: Patient Emotion & Intent Detection 
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
 
#### Task 2 - Summary 
This task demonstrates a targeted pipeline to analyze patient sentiment and intent from a 
mixed-dialogue transcript. 
The approach is two-fold: 

1. Patient Dialogue Extraction: First, the pipeline isolates the patient's voice. A regex 
function extracts all lines spoken by the "Patient:" to create a clean text block. This 
crucial step prevents the physician's dialogue from skewing the analysis. 

2. Flexible Zero-Shot Classification: The extracted text is then analyzed using a 
facebook/bart-large-mnli zero-shot model. This model is used because it can classify 
text against custom, domain-specific labels without any fine-tuning. The model is run 
twice: once to determine the best fit from the sentiment_labels ("Anxious," "Neutral," 
"Reassured") and again to find the intent_labels ("Reporting symptoms," etc.). 
This solution provides a nuanced understanding of the patient's perspective and is highly 
adaptable, as the classification labels can be changed at any time to meet new requirements. 
 
 
### How to Run 
First-Time Setup: When running the model locally for the first time, an internet connection and 
at least 5 GB of data are required to download the necessary model files. 
Folder Structure 

Emitrr_Physician_Notetaker_VishalMaurya/ 
│ 

├── README.md 

├── requirements.txt 

├── Emitrr_Physician_Notetaker_Task_1.ipynb 

└── Emitrr_Physician_Notetaker_Task_2.ipynb 

### Installation & Execution (Bash/Powershell/cmd) 

Bash 

Task 1 

pip install -r requirements.txt 

python emitrr_physician_notetaker_task_1.py 

Or directly run in Colab 

https://colab.research.google.com/drive/18RXhF_xlsMZL62Ka9SjqaYMTAGeTdz1G#scrollTo=z1a
_cP9Fx2zf 

Task 2 

python emitrr_physician_notetaker_task_2.py 

Or directly run in Colab 

https://colab.research.google.com/drive/1GM5nmA-licVUxraTKvDLIKrdhFaTn3s2 


#### Requirements (summary) 

• flask 

• transformers 

• torch 

• sentencepiece 

• accelerate 

• keybert 

• numpy 
 
#### Output 

• Task 1: Structured JSON + formatted medical report 

• Task 2: Sentiment & Intent JSON output 
 
Author 

Vishal Maurya  

vishal.gusknp2022@gmail.com  

National Institute of Technology, Jamshedpur 
 
