import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from tqdm import tqdm

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("/Users/niteshnirranjan/Downloads/data1.csv")

# Ensure only 'tweet' column is used
if 'tweets' in df.columns:
    df = df.rename(columns={'tweets': 'tweet'})

df = df.dropna(subset=['tweet'])
print("Loaded dataset with", len(df), "tweets")

# ---------------------------
# Load pre-trained model
# ---------------------------
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ---------------------------
# Predict sentiment for each tweet
# ---------------------------
sentiments = []
scores = []

for tweet in tqdm(df['tweet'].tolist(), desc="Predicting"):
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    label = "POSITIVE" if pred_label == 1 else "NEGATIVE"
    sentiments.append(label)
    scores.append(confidence)

df['sentiment'] = sentiments
df['confidence'] = scores

# ---------------------------
# Save results
# ---------------------------
output_path = "/Users/niteshnirranjan/Downloads/tweet_sentiment_results.csv"
df.to_csv(output_path, index=False)

print("\nSentiment prediction completed!")
print("Results saved to:", output_path)
print(df.head())

# ---------------------------
# Test custom prediction
# ---------------------------
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()
    return {
        "text": text,
        "sentiment": "POSITIVE" if pred_label == 1 else "NEGATIVE",
        "confidence": confidence
    }

print("\nExample:", predict_sentiment("I love how smooth this experience is!"))
