import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_MAP = {0:"negative", 1:"netral", 2: "positive"}

class RobertaModel:
    def __init__(self, model_dir="models/roberta"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        label_idx = int(np.argmax(probs))
        return LABEL_MAP[label_idx], float(probs[label_idx])
