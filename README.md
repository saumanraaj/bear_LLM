
# BEAR RAG Assistant

This repository implements a Retrieval-Augmented Generation (RAG) pipeline for integrating a large language model (LLM) into the BEAR (Bayesian Experimental Autonomous Researcher) system. The pipeline enables the LLM to answer grounded, domain-specific questions using embedded knowledge from BEAR-related research papers and the GCS GitHub codebase, without requiring model fine-tuning.

## Overview

The RAG system functions as a static memory layer for the LLM, allowing it to simulate long-term understanding of project context. All relevant documents are parsed, chunked, embedded, and stored in a FAISS vector index. At query time, the system retrieves the most relevant content and appends it to the LLM prompt to produce accurate, context-aware responses.

## Technologies Used

- LlamaIndex
- FAISS
- OpenAI GPT-4 Turbo
- PyMuPDF (`fitz`)
- Python 3.10+
- Conda (for environment management)

## Project Structure

```

project/
├── rag\_pipeline.py              # Main pipeline script
├── filtered\_loader.py           # Custom PDF reader with page filtering
├── gcs/                         # Cloned GitHub repository with GCS code
├── papers/                      # BEAR-related PDF documents
└── README.md

````

## Data Sources

### Research Papers

The following BEAR-related documents are embedded in the pipeline:

- A Bayesian Experimental Autonomous Researcher for Mechanical Design
- Superlative Mechanical Energy Absorbing Efficiency...
- lab1.pdf
- lab2.pdf

### GCS Codebase

The pipeline also includes all `.py`, `.md`, and `.txt` files from the [GCS GitHub repository](https://github.com/bu-shapelab/gcs).

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bear_LLM.git
cd bear_LLM
````

### 2. Set up the environment

```bash
conda create -n bear_rag python=3.10
conda activate bear_rag
pip install -r requirements.txt
```

### 3. Set the OpenAI API Key

You can either set this as an environment variable:

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

Or modify the `rag_pipeline.py` script to include it directly.

### 4. Run the pipeline

```bash
python rag_pipeline.py
```

You should see:

```
Loaded 4 documents from papers.
Loaded 27 documents from GitHub repo.
Parsed into 175 text chunks.
Vector index built.
Query engine ready.
```

You may now interactively query the system.

## Sample Questions

* What is SHAP and how is it used in the BEAR framework?
* What does the generate\_gcs.py file do?
* What are the c4 and c8 parameters in the GCS shell design?
* What is the purpose of the BEAR system and how does it select experiments?

## Features

* Domain-specific question answering grounded in internal documents
* Modular and update-friendly—new papers or code can be added easily
* Avoids hallucination by using document context
* No fine-tuning required

## Future Work

* Integration of dynamic experiment datasets
* Visualization tools for STL-based outputs
* Web-based or Slack interface
* Agent-based task chaining and explanation modules

## Contact

Developed by **Sauman Raaj**
Contact: [sauman@bu.edu](mailto:sauman@bu.edu)
