import argparse
import json
import re
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

argp = argparse.ArgumentParser(
    description="Analyze hypothesis length effects from eval outputs."
)
argp.add_argument(
    "--challenge",
    action="store_true",
    help="Use eval_output_challenge and write plots under plots_challenge.",
)
args = argp.parse_args()

base_dir = Path(__file__).resolve().parent
output_dir_name = "eval_output_challenge" if args.challenge else "eval_output"
plots_root = "plots_challenge" if args.challenge else "plots"
path = base_dir / output_dir_name / "eval_predictions.jsonl"
plots_dir = base_dir / plots_root / "artifact_3_short_hypotheses"
plots_dir.mkdir(parents=True, exist_ok=True)

id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
labels = ["entailment", "neutral", "contradiction"]


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
        rows.append(
            {
                "length": length,
                "gold": id2label[ex["label"]],
                "pred": id2label[ex["predicted_label"]],
            }
        )

if len(rows) == 0:
    raise SystemExit("No hypotheses found. Check eval_predictions.jsonl.")

df = pd.DataFrame(rows)
df["correct"] = df["gold"] == df["pred"]
df["bucket"] = df["length"].apply(bucket)

print(f"Loaded {len(df)} predictions from {path}")
print("\nHypothesis length stats (tokens):")
print(df["length"].describe(percentiles=[0.25, 0.5, 0.75]).to_string())

overall_accuracy = df["correct"].mean()
print(f"\nOverall accuracy: {overall_accuracy:.2%}")

bucket_counts = (
    df["bucket"].value_counts().reindex(["short", "medium", "long"]).fillna(0).astype(int)
)
print("\nBucket counts (by hypothesis length):")
print(bucket_counts.to_string())

acc_by_bucket = df.groupby("bucket")["correct"].mean().reindex(["short", "medium", "long"])
print("\nAccuracy by length bucket:")
print(acc_by_bucket.to_string(float_format=lambda x: f"{x:.2%}"))

pred_dist = (
    df.groupby(["bucket", "pred"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=labels, fill_value=0)
    .reindex(index=["short", "medium", "long"], fill_value=0)
)
print("\nPrediction counts by bucket:")
print(pred_dist.to_string())

pred_dist_norm = pred_dist.div(pred_dist.sum(axis=1).replace(0, 1), axis=0)
print("\nPrediction distribution by bucket (share within bucket):")
print(pred_dist_norm.to_string(float_format=lambda x: f"{x:.2f}"))

# Plots
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["length"], bins=20, color="steelblue", edgecolor="white")
ax.axvline(4, color="darkorange", linestyle="--", label="short/medium cutoff")
ax.axvline(8, color="firebrick", linestyle="--", label="medium/long cutoff")
ax.set_xlabel("Hypothesis length (tokens)")
ax.set_ylabel("Example count")
ax.set_title("Distribution of hypothesis lengths")
ax.legend()
plt.tight_layout()
hist_path = plots_dir / "hypothesis_length_histogram.png"
plt.savefig(hist_path, dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
acc_by_bucket.plot(kind="bar", color="seagreen", ylim=(0, 1), ax=ax)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Length bucket")
ax.set_title("Accuracy by hypothesis length")
plt.tight_layout()
acc_plot_path = plots_dir / "hypothesis_length_accuracy.png"
plt.savefig(acc_plot_path, dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
pred_dist_norm[labels].plot(kind="bar", stacked=True, ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel("Share within bucket")
ax.set_xlabel("Length bucket")
ax.set_title("Prediction distribution by hypothesis length")
ax.legend(title="Predicted label", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
pred_plot_path = plots_dir / "hypothesis_length_prediction_distribution.png"
plt.savefig(pred_plot_path, dpi=200)
plt.close()

print("\nPlots saved:")
print(f"- {hist_path}")
print(f"- {acc_plot_path}")
print(f"- {pred_plot_path}")
