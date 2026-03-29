"""
train_error_detector.py — Fine-tune InLegalBERT for legal error detection.

This script fine-tunes the InLegalBERT model (law-ai/InLegalBERT) as a
token classifier to detect spelling, grammatical, and semantic errors
in Indian legal documents.

Architecture:
    InLegalBERT (encoder)
        ↓
    Token Classification Head (linear layer: hidden_size → num_labels)
        ↓
    Per-token labels: O / B-SPELL / I-SPELL / B-GRAM / I-GRAM / B-SEM / I-SEM

Training time estimates:
    GPU (T4/V100):  ~2 hours for 5 epochs on 10,000 examples
    CPU only:       ~10 hours for 5 epochs on 10,000 examples

Usage:
    # Generate data first:
    python scripts/generate_training_data.py

    # Then train (with GPU):
    python scripts/train_error_detector.py

    # On CPU only:
    python scripts/train_error_detector.py --no-fp16
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

# ── CONFIG ─────────────────────────────────────────────────────────────────
BASE_MODEL    = "law-ai/InLegalBERT"   # Pre-trained legal BERT from HuggingFace
OUTPUT_DIR    = "models/nyayai-error-detector"
DATA_DIR      = Path("data/training")
# MAX_LEN       = 256     # Max tokens per window; InLegalBERT supports up to 512
# BATCH_SIZE    = 16      # Reduce to 8 if you get CUDA out-of-memory errors
# EPOCHS        = 5       # 5 epochs is standard for fine-tuning BERT
LEARNING_RATE = 2e-5    # Standard BERT fine-tuning LR
MAX_LEN       = 256     
BATCH_SIZE    = 4       # Optimized for 6GB VRAM
GRAD_ACCUM    = 4       # Simulates a Batch Size of 16 (4 * 4)
EPOCHS        = 3       # 3 is usually plenty for legal fine-tuning
# IOB label scheme
LABEL_LIST = ["O", "B-SPELL", "I-SPELL", "B-GRAM", "I-GRAM", "B-SEM", "I-SEM"]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}
# ────────────────────────────────────────────────────────────────────────────


def load_json_dataset(path: Path):
    """
    Load a JSONL dataset file using a streaming generator.

    JSONL format: one JSON object per line {"tokens": [...], "labels": [...]}
    The generator yields one example at a time — the entire file is NEVER
    loaded into RAM, which prevents OOM on large datasets.

    Teaching note (why json.load() causes OOM):
        json.load() reads the ENTIRE file into a Python object first.
        A 1.3GB JSON array = ~1.3GB of Python dicts in heap memory.
        Plus the Dataset copies it, so you need ~3-4GB RAM just to load data.
        JSONL + generator = constant memory usage regardless of file size.
    """
    from datasets import Dataset

    # Support both .jsonl (new) and legacy .json (array format) files
    jsonl_path = path.with_suffix(".jsonl")
    actual_path = jsonl_path if jsonl_path.exists() else path

    def gen():
        with open(actual_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    yield {"tokens": entry["tokens"], "labels": entry["labels"]}
                except (json.JSONDecodeError, KeyError):
                    continue  # skip malformed lines

    return Dataset.from_generator(gen)


def tokenize_and_align_labels(examples, tokenizer):
    """
    BERT uses wordpiece tokenization — it may split one word into multiple
    subword tokens. For example: "imprisoned" → ["im", "##prison", "##ed"]

    We need to align our word-level labels to subword tokens.
    Standard approach:
        - Label the FIRST subword with the word's label
        - Mark ALL OTHER subwords with -100 (ignored by the loss function)

    Teaching note:
        This alignment step is the most confusing part of token classification.
        The key insight is that we ONLY want the model to predict for the first
        subword of each word — the continuations don't carry independent meaning.
        We use -100 as the ignore index because PyTorch's CrossEntropyLoss
        automatically skips any target labeled -100.
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LEN,
        is_split_into_words=True,  # input is already a list of words
        padding="max_length",
    )

    all_labels = []
    for i, label_seq in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned_labels = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens: [CLS], [SEP], [PAD]
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                # First subword of this word → use the word's label
                aligned_labels.append(LABEL2ID[label_seq[word_id]])
            else:
                # Continuation subword → ignore in loss
                aligned_labels.append(-100)
            prev_word_id = word_id

        all_labels.append(aligned_labels)

    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics(eval_pred):
    """
    Use seqeval for proper span-level F1 scoring.

    Teaching note on seqeval vs sklearn F1:
        sklearn F1 counts individual tokens. seqeval counts spans.
        A "B-SPELL" token followed by "I-SPELL" is ONE spelling error span.
        If the model gets "B-SPELL" right but "I-SPELL" wrong, seqeval
        penalizes this as a missed span. This makes seqeval much stricter
        and more realistic for real error detection quality measurement.
    """
    import numpy as np
    try:
        from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
    except ImportError:
        # Fallback if seqeval not installed
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        correct = ((preds == labels) & (labels != -100)).sum()
        total = (labels != -100).sum()
        return {"accuracy": correct / total}

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Decode predictions: skip -100 (padding/continuation) tokens
    true_labels = [
        [ID2LABEL[l] for l in label_row if l != -100]
        for label_row in labels
    ]
    pred_labels = [
        [ID2LABEL[p] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(preds, labels)
    ]

    return {
        "f1":        f1_score(true_labels, pred_labels, zero_division=0),
        "precision": precision_score(true_labels, pred_labels, zero_division=0),
        "recall":    recall_score(true_labels, pred_labels, zero_division=0),
    }


def train(use_fp16: bool = True):
    """
    Main training function. Loads data, tokenizes, fine-tunes InLegalBERT,
    saves the best model checkpoint.
    """
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            TrainingArguments,
            Trainer,
            DataCollatorForTokenClassification,
            EarlyStoppingCallback,
        )
        import torch
    except ImportError:
        raise ImportError(
            "Install training dependencies: pip install transformers torch datasets seqeval"
        )

    # Check for training data
    train_path = DATA_DIR / "train.json"
    val_path   = DATA_DIR / "val.json"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run: python scripts/generate_training_data.py"
        )

    # ── Load tokenizer and model ──────────────────────────────────────────
    print(f"[INFO] Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        use_safetensors=True,
        ignore_mismatched_sizes=True,  # OK: we're adding a new classification head
    )

    # ── Load and tokenize datasets ────────────────────────────────────────
    print("[INFO] Loading training data...")
    raw_train = load_json_dataset(train_path)
    raw_val   = load_json_dataset(val_path)

    fn = lambda x: tokenize_and_align_labels(x, tokenizer)
    train_ds = raw_train.map(fn, batched=True, remove_columns=raw_train.column_names)
    val_ds   = raw_val.map(fn, batched=True, remove_columns=raw_val.column_names)

    print(f"[OK] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Training arguments ────────────────────────────────────────────────
    use_gpu_fp16 = use_fp16 and torch.cuda.is_available()
    if not torch.cuda.is_available():
        print("[WARN] No GPU detected — training on CPU. This will be slow.")
        print("       Consider using Google Colab with a T4 GPU (free tier).")

    # args = TrainingArguments(
    #     output_dir=OUTPUT_DIR,

    #     # Training schedule
    #     num_train_epochs=EPOCHS,
    #     per_device_train_batch_size=BATCH_SIZE,
    #     per_device_eval_batch_size=BATCH_SIZE,

    #     # Optimizer settings
    #     learning_rate=LEARNING_RATE,
    #     weight_decay=0.01,             # L2 regularization to prevent overfitting
    #     warmup_ratio=0.1,              # Warm up LR for first 10% of steps

    #     # Evaluation and saving
    #     eval_strategy="epoch",         # Evaluate after each epoch
    #     save_strategy="epoch",         # Save after each epoch
    #     load_best_model_at_end=True,   # Use the epoch with best val F1
    #     metric_for_best_model="f1",
    #     greater_is_better=True,

    #     # Speed
    #     fp16=use_gpu_fp16,             # Float16 on GPU halves memory usage and speeds up by ~2x

    #     # Logging
    #     logging_dir="logs/",
    #     logging_steps=50,
    #     report_to="none",              # Disable wandb/tensorboard (add later if needed)

    #     # Reproducibility
    #     seed=42,
    #     data_seed=42,
    # )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # --- MEMORY TUNING FOR RTX 4050 ---
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        fp16=use_gpu_fp16,

        # --- TRAINING FLOW ---
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,

        # --- EVALUATION & SAVING ---
        # Save once per epoch (not every 100 steps) to avoid filling disk.
        # Each checkpoint is ~1.5GB; saving every 100 steps across 2250 steps = 30+ GB!
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,          # Keep only the 1 best checkpoint on disk
        load_best_model_at_end=True,
        metric_for_best_model="f1",

        # --- SYSTEM STABILITY ---
        dataloader_num_workers=0,    # Prevents multi-process OOM
        report_to="none",
        logging_steps=10,
        seed=42,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        # Stops training if val F1 doesn't improve for 2 consecutive epochs
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training NyayAI Error Detector")
    print(f"  Base model:    {BASE_MODEL}")
    print(f"  Output:        {OUTPUT_DIR}")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Device:        {'GPU + FP16' if use_gpu_fp16 else 'CPU'}")
    print(f"  Train samples: {len(train_ds)}")
    print(f"{'='*60}\n")

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n[OK] Model saved to {OUTPUT_DIR}/")
    print(f"     Use with: ErrorDetector(model_path='{OUTPUT_DIR}')")

    # ── Final evaluation ──────────────────────────────────────────────────
    test_path = DATA_DIR / "test.json"
    if test_path.exists():
        print("\n[INFO] Running final evaluation on test set...")
        raw_test = load_json_dataset(test_path)
        test_ds = raw_test.map(fn, batched=True, remove_columns=raw_test.column_names)
        results = trainer.evaluate(test_ds)
        print(f"\n{'='*60}")
        print("Test Set Results:")
        for k, v in results.items():
            print(f"  {k:<20}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune InLegalBERT for error detection")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable float16 (use if running on CPU only)")
    args = parser.parse_args()
    train(use_fp16=not args.no_fp16)
