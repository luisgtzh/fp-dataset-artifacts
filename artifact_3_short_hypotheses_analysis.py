import json
import re
from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parent / "eval_output" / "eval_predictions.jsonl"

id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def bucket(n):
    if n < 4:
        return "short"
    elif n < 8:
        return "medium"
    return "long"

rows = []

with open(path, "r") as f:
    for line in f:
        ex = json.loads(line)
        hyp = ex["hypothesis"]
        length = len(tokenize(hyp))
        rows.append({
            "length": length,
            "gold": id2label[ex["label"]],
            "pred": id2label[ex["predicted_label"]]
        })

df = pd.DataFrame(rows)
df["correct"] = df["gold"] == df["pred"]
df["bucket"] = df["length"].apply(bucket)

pred_dist = df.groupby(["bucket", "pred"]).size().unstack(fill_value=0)
print(pred_dist)

pred_dist_norm = pred_dist.div(pred_dist.sum(axis=1), axis=0)
print(pred_dist_norm)

print(df.groupby("bucket")["correct"].mean())