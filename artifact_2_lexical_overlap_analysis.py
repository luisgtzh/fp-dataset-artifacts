import json
import re
from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parent / "eval_output" / "eval_predictions.jsonl"

id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def bucket(n):
    if n < 0.3:
        return "low"
    elif n < 0.7:
        return "medium"
    return "high"

records = []

with open(path, "r") as f:
    for line in f:
        ex = json.loads(line)

        prem_tokens = set(tokenize(ex["premise"]))
        hyp_tokens = set(tokenize(ex["hypothesis"]))

        if len(hyp_tokens) == 0:
            continue

        overlap_ratio = len(prem_tokens & hyp_tokens) / len(hyp_tokens)

        records.append({
            "overlap": overlap_ratio,
            "gold": id2label[ex["label"]],
            "pred": id2label[ex["predicted_label"]]
        })

df = pd.DataFrame(records)
df["correct"] = df["gold"] == df["pred"]
df["bucket"] = df["overlap"].apply(bucket)

print("Accuracy by overlap bucket:")
print(df.groupby("bucket")["correct"].mean())

pred_dist = df.groupby(["bucket", "pred"]).size().unstack(fill_value=0)
print("Prediction counts by bucket:")
print(pred_dist)

pred_dist_norm = pred_dist.div(pred_dist.sum(axis=1), axis=0)
print("Prediction distribution by bucket:")
print(pred_dist_norm)

subset = df[(df["bucket"] == "high") & (df["gold"] != "entailment")]

rate = (subset["pred"] == "entailment").mean()

print("Fraction of predictions = entailment on high-overlap NON-entailment examples:")
print(rate)