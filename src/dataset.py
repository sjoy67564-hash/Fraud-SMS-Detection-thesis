import pandas as pd
import torch
import re
from torch.utils.data import Dataset

class SMSDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):

        self.df = pd.read_csv(
            csv_path,
            quotechar='"',
            on_bad_lines='skip',
            engine='python'
        )

        # ---------------- CLEAN TEXT ----------------
        def clean(text):
            text = str(text).lower()
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"[^a-zA-Z0-9\u0980-\u09FF ]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        # column fix
        if "message" in self.df.columns:
            self.df["text"] = self.df["message"]
        elif "text" in self.df.columns:
            self.df["text"] = self.df["text"]
        else:
            raise ValueError("CSV must have 'message' or 'text' column")

        self.df["text"] = self.df["text"].apply(clean)

        # ---------------- LABEL ----------------
        self.df["label"] = self.df["label"].astype(str).str.strip().str.lower()

        self.label_map = {
            "legit": 0,
            "fraud": 1
        }

        self.df = self.df[self.df["label"].isin(self.label_map.keys())]
        self.df = self.df.drop_duplicates()
        self.df = self.df.reset_index(drop=True)

        self.tokenizer = tokenizer
        self.max_len = max_len

        # 🔥 IMPORTANT FIX: tokenize ONCE (NOT inside __getitem__)
        self.encodings = tokenizer(
            self.df["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=self.max_len
        )

        self.labels = [
            self.label_map[l] for l in self.df["label"].tolist()
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

        return item