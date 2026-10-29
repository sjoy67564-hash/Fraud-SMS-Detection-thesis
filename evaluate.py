import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Subset

from src.dataset import SMSDataset

print(" Evaluation Started...")


model_path = "./models/saved_model_augmented/final"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


csv_path = r"E:\Fraud_SMS_Detection_Project\data\fraud_sms_augmented.csv"

dataset = SMSDataset(csv_path, tokenizer)


dataset = Subset(dataset, range(min(5000, len(dataset))))

loader = DataLoader(dataset, batch_size=16)

y_true = []
y_pred = []


with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs.logits, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())


precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("FINAL RESULTS")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1-Score :", round(f1 * 100, 2), "%")