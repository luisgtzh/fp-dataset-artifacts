import json
import re
from collections import defaultdict
from pathlib import Path

path = Path(__file__).resolve().parent / "eval_output" / "eval_predictions.jsonl"

id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def normalize(count_dict):
    norm = {}
    for w, labels in count_dict.items():
        total = sum(labels.values())
        if total == 0:
            continue
        norm[w] = {lbl: labels[lbl] / total for lbl in labels}
    return norm

neg_words = ["not", "nobody", "never", "no", "nothing"]

gold_counts = {w: defaultdict(int) for w in neg_words}
pred_counts = {w: defaultdict(int) for w in neg_words}

with open(path, "r") as f:
    for line in f:
        ex = json.loads(line)

        hyp = ex["hypothesis"]
        hyp_tokens = set(tokenize(hyp))

        gold = id2label[ex["label"]]
        pred = id2label[ex["predicted_label"]]

        for w in neg_words:
            if w in hyp_tokens:
                gold_counts[w][gold] += 1
                pred_counts[w][pred] += 1

gold_probs = normalize(gold_counts)
pred_probs = normalize(pred_counts)

print("Gold distributions for negation words:")
for w, dist in gold_probs.items():
    print(w, dist)

print("\nPredicted distributions for negation words:")
for w, dist in pred_probs.items():
    print(w, dist)