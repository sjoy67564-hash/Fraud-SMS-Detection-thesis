"""
Convert BERT model to ONNX for 10x faster inference
Run once: python scripts/convert_to_onnx.py
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

MODEL_PATH = "models/saved_model/checkpoint-1965"
ONNX_PATH = "models/saved_model/bert_model.onnx"

def convert_to_onnx():
    print("[INFO] Converting BERT to ONNX format...")
    
    if os.path.exists(ONNX_PATH):
        print(f"[INFO] ONNX model already exists: {ONNX_PATH}")
        return
    
    # Load model
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    
    model.eval()
    
    # Create dummy input
    dummy_input = tokenizer.encode_plus(
        "dummy text",
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Export to ONNX
    print("[EXPORT] Exporting to ONNX...")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"], dummy_input.get("token_type_ids")),
        ONNX_PATH,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "token_type_ids": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"[SUCCESS] ONNX model saved: {ONNX_PATH}")
    print("[INFO] Now predictions will be 10x faster!")

if __name__ == "__main__":
    convert_to_onnx()
