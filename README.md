# Sentiment Analysis using DistilBERT

## Overview
This project performs **sentiment analysis** on tweet data using the **DistilBERT transformer model**, fine-tuned on the SST-2 dataset.  
The goal is to classify each tweet as **positive** or **negative**, along with a confidence score.  
The pipeline handles preprocessing, prediction, and result export — all implemented in **PyTorch** using **Hugging Face Transformers**.

---

## Model Used
- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`  
- **Framework:** PyTorch  
- **Tokenizer:** DistilBERT Fast Tokenizer  
- **Source:** Hugging Face Transformers library  

This pre-trained DistilBERT model has been fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset for binary sentiment classification.

---

## Dataset
The input dataset is a CSV file containing a column named `tweet`.  
If your file uses a different column name like `tweets`, the script automatically renames it.  
Example format:

| tweet |
|--------|
| I love this product! |
| This is terrible  |
| The movie was amazing! |

---

## Requirements
Install the required dependencies:
```bash
pip install pandas torch transformers tqdm

## Project Structure
.
├── sentiment_analysis.py
├── data1.csv
└── tweet_sentiment_results.csv
