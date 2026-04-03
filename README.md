# CS 5170 Project 2: Retrieval-Augmented Generation (RAG)

A retrieval-augmented generation system for factual question answering about Carnegie Mellon University and the Language Technologies Institute (LTI).

## Team

- Eduardo Gaxiola
- Mihail Chitorog (mikeleon001)
- Sunjay Guttikonda

## Domain

**CMU / LTI** — Questions about LTI faculty, research papers, CMU course schedules, academic calendars, university history, and program handbooks.

## System Overview

Given an input question, the system:
1. Retrieves the most relevant documents from the CMU/LTI knowledge base
2. Passes the question + retrieved context to OLMo 2 to generate a concise answer

Two variants are implemented:
- **Baseline**: BM25 sparse retrieval + OLMo 2 generator
- **Improved**: Dense retrieval (`sentence-transformers`) + OLMo 2 generator

## Repo Structure

```
cs5170-project2/
├── data/
│   ├── test/
│   │   ├── questions.txt           # Our annotated test questions
│   │   └── reference_answers.txt   # Reference answers for evaluation
│   └── train/
│       ├── questions.txt           # Training questions
│       └── reference_answers.txt   # Training reference answers
├── src/
│   ├── collect_data.py     # Scraping & data collection scripts
│   ├── build_index.py      # Build BM25 and dense retrieval indexes
│   ├── rag_pipeline.py     # Main RAG inference pipeline
│   └── evaluate.py         # Evaluation (EM, F1, recall)
├── docs/
│   └── report.pdf          # Final report (added before submission)
├── system_outputs/
│   ├── system_output_1.txt # BM25 + OLMo 2 outputs
│   └── system_output_2.txt # Dense retrieval + OLMo 2 outputs
├── contributions.md
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/mikeleon001/cs5170-project2.git
cd cs5170-project2
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

### 1. Collect data
```bash
python src/collect_data.py
```

### 2. Build retrieval index
```bash
python src/build_index.py
```

### 3. Run RAG inference
```bash
python src/rag_pipeline.py --questions data/test/questions.txt --output system_outputs/system_output_1.txt --retriever bm25
python src/rag_pipeline.py --questions data/test/questions.txt --output system_outputs/system_output_2.txt --retriever dense
```

### 4. Evaluate
```bash
python src/evaluate.py --output system_outputs/system_output_1.txt --reference data/test/reference_answers.txt
```

## Models

- **Generator**: [allenai/OLMo-2-1124-7B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)
- **Dense Retriever**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## References

- Lewis et al., 2021. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- Touvron et al., 2023. [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
