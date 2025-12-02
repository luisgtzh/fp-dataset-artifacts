import argparse
import json
import re
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

argp = argparse.ArgumentParser(
    description="Analyze lexical overlap performance from eval outputs."
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
plots_dir = base_dir / plots_root / "artifact_2_lexical_overlap"
plots_dir.mkdir(parents=True, exist_ok=True)

id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
labels = ["entailment", "neutral", "contradiction"]


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

        records.append(
            {
                "overlap": overlap_ratio,
                "gold": id2label[ex["label"]],
                "pred": id2label[ex["predicted_label"]],
            }
        )

if len(records) == 0:
    raise SystemExit("No evaluatable rows found. Check eval_predictions.jsonl contents.")

df = pd.DataFrame(records)
df["correct"] = df["gold"] == df["pred"]
df["bucket"] = df["overlap"].apply(bucket)

print(f"Loaded {len(df)} predictions from {path}")
print("\nOverlap distribution stats:")
print(df["overlap"].describe(percentiles=[0.25, 0.5, 0.75]).to_string())

overall_accuracy = df["correct"].mean()
print(f"\nOverall accuracy: {overall_accuracy:.2%}")

bucket_counts = (
    df["bucket"].value_counts().reindex(["low", "medium", "high"]).fillna(0).astype(int)
)
print("\nBucket counts (by overlap ratio):")
print(bucket_counts.to_string())

acc_by_bucket = df.groupby("bucket")["correct"].mean().reindex(["low", "medium", "high"])
print("\nAccuracy by overlap bucket:")
print(acc_by_bucket.to_string(float_format=lambda x: f"{x:.2%}"))

pred_dist = (
    df.groupby(["bucket", "pred"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=labels, fill_value=0)
    .reindex(index=["low", "medium", "high"], fill_value=0)
)
print("\nPrediction counts by bucket:")
print(pred_dist.to_string())

pred_dist_norm = pred_dist.div(pred_dist.sum(axis=1).replace(0, 1), axis=0)
print("\nPrediction distribution by bucket (share within bucket):")
print(pred_dist_norm.to_string(float_format=lambda x: f"{x:.2f}"))

subset = df[(df["bucket"] == "high") & (df["gold"] != "entailment")]
rate = (subset["pred"] == "entailment").mean() if len(subset) else 0.0
print(
    "\nFraction of predictions = entailment on high-overlap NON-entailment examples "
    f"(n={len(subset)}): {rate:.2%}"
)

# Plots
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["overlap"], bins=20, color="steelblue", edgecolor="white")
ax.axvline(0.3, color="darkorange", linestyle="--", label="low/medium cutoff")
ax.axvline(0.7, color="firebrick", linestyle="--", label="medium/high cutoff")
ax.set_xlabel("Hypothesis-premise overlap ratio")
ax.set_ylabel("Example count")
ax.set_title("Distribution of lexical overlap")
ax.legend()
plt.tight_layout()
hist_path = plots_dir / "lexical_overlap_histogram.png"
plt.savefig(hist_path, dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
acc_by_bucket.plot(kind="bar", color="seagreen", ylim=(0, 1), ax=ax)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Overlap bucket")
ax.set_title("Accuracy by overlap bucket")
plt.tight_layout()
acc_plot_path = plots_dir / "lexical_overlap_accuracy.png"
plt.savefig(acc_plot_path, dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
pred_dist_norm[labels].plot(kind="bar", stacked=True, ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel("Share within bucket")
ax.set_xlabel("Overlap bucket")
ax.set_title("Prediction distribution by overlap bucket")
ax.legend(title="Predicted label", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
pred_plot_path = plots_dir / "lexical_overlap_prediction_distribution.png"
plt.savefig(pred_plot_path, dpi=200)
plt.close()

print("\nPlots saved:")
print(f"- {hist_path}")
print(f"- {acc_plot_path}")
print(f"- {pred_plot_path}")
