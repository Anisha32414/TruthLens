from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import gdown
import zipfile


# Load DistilBERT model and tokenizer
model_path = os.path.join(os.path.dirname(__file__), ".")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def get_bert_score(text):
    """
    Uses DistilBERT model to predict the authenticity score of the text.
    Returns a score between 0 and 1, where higher values indicate more authentic content.

    Args:
        text (str): The input text to analyze

    Returns:
        float: Probability score between 0 and 1
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

    # Run inference

BASE_DIR = os.path.dirname(__file__)

MODEL_DIR = os.path.join(BASE_DIR, "../bert_model")

ZIP_PATH = os.path.join(BASE_DIR, "bert_model.zip")

FILE_ID = "1-NgDal2jM3q-9vJWl86AZwZnmNcrdY-H"


def download_model():
    """Download model if not present"""

    if not os.path.exists(MODEL_DIR):
        print("Downloading BERT model...")

        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

        print("Extracting model...")

        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(BASE_DIR, ".."))

        print("Model ready!")


download_model()

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def get_bert_score(text):

    if isinstance(text, str):
        texts = [text]
    else:
        texts = text
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits


    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Return the probability of being authentic (class 1)
    # Assuming class 1 represents authentic/real content
    return probabilities[0][1].item()


    return probs[:, 1].tolist() if len(probs) > 1 else probs[0, 1].item()