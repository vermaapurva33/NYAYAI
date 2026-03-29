"""
generate_training_data.py — Build a labeled error-detection dataset for NyayAI.

This script:
1. Loads clean Indian legal text (from HuggingFace datasets or local files)
2. Injects synthetic errors at a controlled rate
3. Labels each token with IOB tags: O, B-SPELL, I-SPELL, B-GRAM, I-GRAM, B-SEM, I-SEM
4. Saves train/val/test splits as JSON

Why synthetic errors?
    We don't have a hand-labeled dataset of "legal PDFs with marked errors".
    Synthetic injection is the standard NLP approach: take clean text, corrupt
    it in realistic ways, and train the model to find what we corrupted.
    The model then generalizes to real errors that follow similar patterns.

Usage:
    python scripts/generate_training_data.py
    # Takes ~10 minutes for 10,000 examples on CPU
"""

import random
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

# ── CONFIG ─────────────────────────────────────────────────────────────────
OUTPUT_DIR    = Path("data/training")
RANDOM_SEED   = 42
ERROR_RATE    = 0.35    # 35% of sentences get an injected error
TRAIN_SPLIT   = 0.8     # 80% train, 10% val, 10% test
VAL_SPLIT     = 0.1
MAX_SENTENCES = 150_000  # Extended cap for Colab training (was 15k)
                         # 15k → ~1.3MB output files, ~2hr training on RTX 4050
# ────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

# IOB label scheme
LABEL_LIST = ["O", "B-SPELL", "I-SPELL", "B-GRAM", "I-GRAM", "B-SEM", "I-SEM"]

# ── SPELLING SUBSTITUTION TABLE ────────────────────────────────────────────
# Common OCR-style and typographic mistakes in Indian legal documents
SPELLING_ERRORS = {
    "accused":        ["accussed",    "accusd",        "acused"],
    "petition":       ["pettion",     "petiton",       "petiiton"],
    "judgment":       ["jugment",     "judgement",     "judgemant"],  # UK vs US also common
    "section":        ["seciton",     "setion",        "secton"],
    "imprisonment":   ["imprisonmant","imprisonement", "imprionment"],
    "respondent":     ["respondant",  "respodent",     "respondent"],
    "magistrate":     ["magistrte",   "magistrait",    "magistate"],
    "appellant":      ["appelant",    "apelant",       "appelant"],
    "jurisdiction":   ["jurisidction","jurisdiciton",  "jurisdction"],
    "honorable":      ["honrable",    "honouable",     "honerable"],
    "advocate":       ["advocte",     "advoate",       "advocaet"],
    "constitution":   ["consitution", "constitiution", "constituton"],
    "defendant":      ["defendent",   "defendat",      "defandant"],
    "acquittal":      ["acquital",    "acquittle",     "acquital"],
    "bail":           ["bial",        "bail",          "bale"],  # mild
    "cognizance":     ["cognisance",  "cognizence",    "cognizanse"],
    "summons":        ["summon",      "sumnons",       "sumons"],
    "witnesses":      ["witneses",    "witnesess",     "witnessess"],
    "affidavit":      ["affidovit",   "affidivit",     "afidavit"],
    "pronounced":     ["pronouced",   "pronouned",     "prononced"],
}

# ── SECTION NUMBER SUBSTITUTION TABLE ─────────────────────────────────────
# Wrong IPC/BNS/CrPC section citations — the most critical semantic errors
SECTION_ERRORS: Dict[str, List[str]] = {
    # IPC (Indian Penal Code) — now BNS (Bharatiya Nyaya Sanhita)
    "302":  ["307", "304", "303"],    # Murder → Attempt/Culpable Homicide
    "307":  ["302", "304", "326"],    # Attempt to murder
    "304":  ["302", "307", "304A"],   # Culpable homicide
    "420":  ["406", "408", "380"],    # Cheating
    "376":  ["377", "354", "509"],    # Rape
    "498A": ["498", "494", "495"],    # Cruelty by husband
    "370":  ["374", "372", "373"],    # Human trafficking
    "326":  ["324", "322", "308"],    # Grievous hurt
    "406":  ["420", "408", "409"],    # Criminal breach of trust
    "354":  ["376", "354A", "509"],   # Assault on woman
    # CrPC (now BNSS)
    "482":  ["483", "378", "407"],    # Inherent powers of High Court
    "438":  ["439", "437", "167"],    # Anticipatory bail
    "439":  ["438", "437", "389"],    # Regular bail by Sessions Court
    "167":  ["173", "170", "169"],    # Remand
    # Constitution
    "21":   ["19", "22", "14"],       # Right to life → other fundamental rights
    "14":   ["19", "21", "226"],      # Equality before law
    "226":  ["227", "32", "136"],     # High Court writ jurisdiction
    "32":   ["226", "136", "142"],    # Supreme Court writ jurisdiction
}


def _inject_spelling_error(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """
    Pick a random content word and replace it with a common misspelling.
    Returns (modified_tokens, labels).
    """
    labels = ["O"] * len(tokens)
    # Find candidates: known words or any long alpha word
    candidates = [
        i for i, t in enumerate(tokens)
        if t.lower() in SPELLING_ERRORS or (len(t) > 5 and t.isalpha())
    ]
    if not candidates:
        return tokens, labels

    idx = random.choice(candidates)
    word = tokens[idx]
    lower = word.lower()

    if lower in SPELLING_ERRORS:
        bad = random.choice(SPELLING_ERRORS[lower])
    else:
        # Generic swap: exchange two adjacent characters in the middle of the word
        chars = list(word)
        i = random.randint(1, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        bad = "".join(chars)

    tokens = tokens.copy()
    tokens[idx] = bad
    labels[idx] = "B-SPELL"
    return tokens, labels


def _inject_semantic_error(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """
    Find a section number token and replace it with a similar but wrong one.
    Returns (modified_tokens, labels).
    """
    labels = ["O"] * len(tokens)
    # Find section number tokens
    section_indices = [
        i for i, t in enumerate(tokens)
        if t in SECTION_ERRORS
    ]
    if not section_indices:
        return tokens, labels

    idx = random.choice(section_indices)
    bad_section = random.choice(SECTION_ERRORS[tokens[idx]])
    tokens = tokens.copy()
    tokens[idx] = bad_section
    labels[idx] = "B-SEM"
    return tokens, labels


def _make_example(sentence: str) -> Dict:
    """
    Decide whether to inject an error and which type.
    Returns a training example dict with tokens and IOB labels.
    """
    tokens = sentence.split()
    if len(tokens) < 5:
        return {"tokens": tokens, "labels": ["O"] * len(tokens)}

    r = random.random()
    if r < ERROR_RATE * 0.6:
        # Inject spelling error (more common than semantic in practice)
        tokens, labels = _inject_spelling_error(tokens)
    elif r < ERROR_RATE:
        # Inject semantic error (wrong section number)
        tokens, labels = _inject_semantic_error(tokens)
    else:
        # Clean example — no error (model must also learn what "correct" looks like)
        labels = ["O"] * len(tokens)

    return {"tokens": tokens, "labels": labels}


def load_sentences_from_huggingface() -> List[str]:
    """
    Load clean Indian legal sentences from a HuggingFace dataset.
    Falls back to a small built-in sample if the library isn't installed.
    """
    try:
        from datasets import load_dataset
        print("[INFO] Loading IL-TUR LSI dataset from HuggingFace...")
        
        # 'lsi' is the primary English subset for statute identification
        ds = load_dataset("Exploration-Lab/IL-TUR", "lsi", trust_remote_code=True)
        sentences = []
        
        # IL-TUR LSI uses 'train', 'validation', and 'test'
        for split_name in ["train", "validation", "test"]:
            if split_name in ds:
                for item in ds[split_name]:
                    # 1. Extract raw data
                    raw_content = item.get("text") or item.get("section_text") or ""
                    
                    # 2. Handle List vs String (The Fix)
                    if isinstance(raw_content, list):
                        # Join the list into one continuous block of text
                        full_text = " ".join(raw_content)
                    else:
                        full_text = str(raw_content)

                    if full_text:
                        # 3. Split into sentences and filter
                        # We use a space after the dot to avoid splitting 'Sec. 144'
                        for sent in full_text.replace(". ", ".\n").split("\n"):
                            sent = sent.strip()
                            # Filtering for quality (8+ words) as you requested
                            if len(sent.split()) >= 8:
                                sentences.append(sent)
                                
        print(f"[OK] Loaded {len(sentences)} sentences from HuggingFace")
        return sentences
    except Exception as e:
        print(f"[WARN] Could not load HuggingFace dataset: {e}")
        print("[INFO] Using built-in sample sentences (small dataset)")
        return _builtin_sample_sentences()


def load_sentences_from_files(corpus_dir: Path) -> List[str]:
    """
    Load sentences from local .txt files in data/legal_corpus/.
    Each line in the file is treated as one sentence.
    """
    sentences = []
    txt_files = list(corpus_dir.rglob("*.txt"))
    print(f"[INFO] Found {len(txt_files)} text files in {corpus_dir}")

    for filepath in txt_files:
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if len(line.split()) >= 8:  # skip very short lines
                        sentences.append(line)
        except Exception as e:
            print(f"[WARN] Could not read {filepath}: {e}")

    print(f"[OK] Loaded {len(sentences)} sentences from local files")
    return sentences


def _builtin_sample_sentences() -> List[str]:
    """
    Minimal built-in sample for testing without any external dependencies.
    Real training needs at least 5,000 sentences for good results.
    """
    return [
        "The accused was found guilty under section 302 of the Indian Penal Code",
        "The learned magistrate took cognizance of the offence punishable under section 420",
        "The appellant filed a petition before the honorable High Court under Article 226",
        "The respondent appeared before the court and filed a written statement",
        "The bail application filed by the accused was dismissed by the sessions court",
        "The judgment was pronounced by the honorable Supreme Court of India",
        "The defendant was acquitted of all charges by the trial court",
        "The witness gave an affidavit before the magistrate under section 167",
        "The accused was granted anticipatory bail under section 438 of CrPC",
        "The summons were issued to the respondent by the learned magistrate",
        "The court exercised its inherent powers under section 482 of CrPC",
        "The accused was sentenced to imprisonment for life under section 302 IPC",
        "The jurisdiction of this court is invoked under Article 32 of the Constitution",
        "The right to life guaranteed under Article 21 cannot be taken away arbitrarily",
        "The prosecution examined nine witnesses during the trial proceedings",
        "The advocates appearing for the petitioner argued the matter at length",
        "The constitution bench comprising five judges delivered the landmark judgment",
        "The accused was charged with offences punishable under section 376 IPC",
        "The sessions court convicted the accused under section 307 of IPC",
        "The high court allowed the revision petition filed by the complainant",
    ] * 50  # Repeat for a slightly larger base dataset


def build_dataset(sentences: List[str], output_dir: Path):
    """
    Convert sentences to training examples and save train/val/test splits.
    Output format: JSONL (one JSON object per line) — enables streaming,
    so the training script doesn't need to load the entire file into RAM.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Generating {len(sentences)} training examples...")
    examples = [_make_example(s) for s in sentences]
    random.shuffle(examples)

    n = len(examples)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)

    splits = {
        "train": examples[:n_train],
        "val":   examples[n_train:n_train + n_val],
        "test":  examples[n_train + n_val:],
    }

    for split_name, data in splits.items():
        # Save as JSONL: one JSON object per line (streamable, memory-efficient)
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        error_count = sum(1 for e in data if any(l != "O" for l in e["labels"]))
        size_mb = out_path.stat().st_size / 1_048_576
        print(f"  [{split_name:>5}] {len(data):>5} examples, "
              f"{error_count:>4} with errors ({error_count/len(data)*100:.1f}%)"
              f" {size_mb:.1f}MB → {out_path}")

    print(f"\n[OK] Dataset saved to {output_dir}/")
    print(f"     Total: {n} | Train: {n_train} | Val: {n_val} | Test: {n - n_train - n_val}")
    print(f"     Format: JSONL (streamable — no OOM during training)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate NyayAI training data")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/legal_corpus"),
                        help="Directory of .txt legal corpus files (optional)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-sentences", type=int, default=MAX_SENTENCES,
                        help=f"Max sentences to use (default: {MAX_SENTENCES})")
    args = parser.parse_args()

    corpus_dir = args.corpus_dir
    if corpus_dir.exists() and any(corpus_dir.rglob("*.txt")):
        sentences = load_sentences_from_files(corpus_dir)
    else:
        sentences = load_sentences_from_huggingface()

    # Cap to avoid OOM during training
    if len(sentences) > args.max_sentences:
        print(f"[INFO] Capping {len(sentences):,} sentences → {args.max_sentences:,} (use --max-sentences to change)")
        random.shuffle(sentences)
        sentences = sentences[:args.max_sentences]

    if len(sentences) < 100:
        print(f"[WARN] Only {len(sentences)} sentences found. "
              "For production-quality training, aim for 10,000+.")

    build_dataset(sentences, args.output_dir)
