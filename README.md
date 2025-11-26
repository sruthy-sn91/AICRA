# AICRA
# AI-Powered Contracts Analyzer (AICRA)
AICRA is an end-to-end, clause-centric system for contract parsing, clause classification, retrieval-augmented question answering (RAG), policy compliance checks, and version comparison.
The pipeline ingests PDF/DOCX/TXT documents, segments them into clauses, labels each clause using a transformer-based classifier, retrieves relevant evidence with FAISS, and generates concise, citation-grounded answers using a LoRA-adapted instruction LLM.
This repository contains the full codebase, trained artifacts, metrics outputs, and Streamlit UI needed to reproduce the system.

# ✨ Key Features

**✔️ Clause-centric analysis**

 - Contract parsing using PyMuPDF / python-docx. 
 - Structural segmentation into clause units. 
 - Transformer-based classifier trained on CUAD clause groups.

**✔️ Retrieval-Augmented Q&A**

 - Sentence-Transformer embeddings for clauses.
 - FAISS (IndexFlatIP) vector search for Top-K evidence retrieval.
 - LoRA-adapted instruction LLM generates evidence-bound answers with clause IDs cited.

**✔️ Policy Studio**

 - Deterministic rules engine supporting:
	 - requires: categories that must exist
	 - forbids: categories that must not exist
	 - rules: include/exclude keyword constraints per category
 - Produces PASS/FAIL compliance report.

**✔️ Compare Versions**

 - Align clauses across two contract versions using embedding
   similarity.
   Reports Added / Removed / Modified clauses.
   Generates an LLM-based change summary.
 
**✔️ Reproducible Experiments**

 - All metrics stored in metrics_train.json, metrics_val.json,
   metrics_test.json.
   Includes classification reports and confusion matrices.
   LoRA adapter dirs saved for all Q&A model variations.

# 📂 Repository Structure
.
├── streamlit_app.py             # Main UI
├── rag_lora_contracts1.py       # LoRA training and inference pipeline
├── clause_classifier.ipynb      # Notebook for training classifier
├── data/                        # Raw and processed CUAD artifacts
│   ├── label_group_xlsx/        # Clause labels
│   ├── cuad-qa/                 # Curated QA subset
│   └── sample_contracts/        # Example documents
├── metrics_train.json
├── metrics_val.json
├── metrics_test.json
├── models_leaderboard_*.csv
├── lora_out_*                   # LoRA adapter directories
├── faiss_index/                 # Stored FAISS index (optional)
├── requirements.txt
└── README.md

# 📦 Installation
1. Clone the repository
	*git clone https://github.com/<your-org>/aicra.git
	cd aicra*
2. Create a conda/venv environment
	*conda create -n aicra python=3.13
	conda activate aicra*
3. Install dependencies
	*pip install -r requirements.txt*

Note for Apple Silicon users
PyTorch MPS is automatically used; bitsandbytes will run in CPU mode.

🚀 Running the Application
Start the Streamlit UI
*streamlit run streamlit_app.py*

This launches the interactive system with three major tabs:

**Contract Analysis**
Upload → Parse → Segment → Classify
View clause table, labels, confidence
Chat over Contract (RAG)
Ask a question
System retrieves Top-K clauses
Generates cited answer

**Policy Studio**
Build or upload JSON rules
Evaluate compliance (PASS/FAIL)

**Compare Versions**
Upload OLD and NEW versions
See Added / Removed / Modified
Generate LLM Change Report

🧠 Training the Clause Classifier
Run the notebook:
jupyter notebook clause_classifier.ipynb

**Outputs:**

Model checkpoint directory
metrics_train/val/test.json

Confusion matrix plots
models_leaderboard_*.csv

# 🔧 Training the LoRA Adapter
Use the training script:
*python rag_lora_contracts1.py train \
    --train_file ./data/cuad_sft_5k.json \
    --base_model <model-name> \
    --out_dir ./lora_out_qwen2_7b_fast \
    --epochs 1*

Outputs include:
LoRA adapter directory
Training logs (loss, grad norm, LR schedule)

# 📊 Results Summary

The classifier shows strong diagonal performance with errors concentrated among adjacent clause families (IP Ownership ↔ Assignment, etc.).

FAISS retrieval is fast (tens of ms) and stable at K=5.
LoRA-adapted LLM produces consistent citations and avoids hallucination when a concept is absent.
End-to-end latency: ~1–3 seconds for a grounded answer on Apple MPS.
Policy Studio and Compare Versions provide deterministic, auditable outputs.
For full details, see the project report’s Results & Analysis chapter.

# 🔍 Datasets

**CUAD (Contract Understanding Atticus Dataset)**
Used for:
Clause classification
QA subset preparation
Examples for segmentation

**CUAD Q&A Subset (curated)**
Used for:
Training LoRA adapters
Grounded Q&A
Note: Only clauses present in the document are used to answer questions; external knowledge is not injected.

# 🔁 Reproducibility
This project saves all metrics and model artifacts required to reproduce results:
metrics_*.json → precision, recall, F1, support
Confusion matrices
Leaderboard CSVs
LoRA adapter directories
FAISS index (optional)
Requirements file with pinned versions

# 🛠️ Tools & Technologies

Python 3.13
PyTorch (MPS acceleration)
Sentence-Transformers
FAISS (IndexFlatIP)
Hugging Face Transformers
PEFT (LoRA)
Streamlit
pandas / numpy
PyMuPDF / python-docx
