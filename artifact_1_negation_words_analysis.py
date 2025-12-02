import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

argp = argparse.ArgumentParser(
    description="Analyze negation word performance from eval outputs."
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
plots_dir = base_dir / plots_root / "artifact_1_negation_words"
plots_dir.mkdir(parents=True, exist_ok=True)

id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
labels = ["entailment", "neutral", "contradiction"]


def tokenize(text):
    return re.findall(r"\w+", text.lower())


neg_words = ["not", "nobody", "never", "no", "nothing"]

gold_counts = {w: defaultdict(int) for w in neg_words}
pred_counts = {w: defaultdict(int) for w in neg_words}
rows = []
total_examples = 0
examples_with_negation = 0

with open(path, "r") as f:
    for line in f:
        total_examples += 1
        ex = json.loads(line)

        hyp_tokens = set(tokenize(ex["hypothesis"]))

        gold = id2label[ex["label"]]
        pred = id2label[ex["predicted_label"]]

        has_negation = False
        for w in neg_words:
            if w in hyp_tokens:
                has_negation = True
                gold_counts[w][gold] += 1
                pred_counts[w][pred] += 1
                rows.append({"word": w, "gold": gold, "pred": pred})
        if has_negation:
            examples_with_negation += 1

if total_examples == 0:
    raise SystemExit(f"No examples found in {path}")

if len(rows) == 0:
    raise SystemExit("No negation words found in hypotheses. Nothing to report.")

df = pd.DataFrame(rows)
df["correct"] = df["gold"] == df["pred"]

gold_counts_df = pd.DataFrame(
    [{**{"word": w}, **{lbl: gold_counts[w][lbl] for lbl in labels}} for w in neg_words]
).set_index("word")
pred_counts_df = pd.DataFrame(
    [{**{"word": w}, **{lbl: pred_counts[w][lbl] for lbl in labels}} for w in neg_words]
).set_index("word")

gold_dist_df = gold_counts_df.div(gold_counts_df.sum(axis=1).replace(0, 1), axis=0)
pred_dist_df = pred_counts_df.div(pred_counts_df.sum(axis=1).replace(0, 1), axis=0)

acc_by_word = df.groupby("word")["correct"].mean().reindex(neg_words)
overall_accuracy = df["correct"].mean()

print(f"Loaded {total_examples} predictions from {path}")
print(
    f"Examples containing any tracked negation word: {examples_with_negation} "
    f"({examples_with_negation / total_examples:.1%})"
)
print(f"Overall accuracy on negation-containing examples: {overall_accuracy:.2%}")

coverage = gold_counts_df.sum(axis=1).to_frame(name="examples")
print("\nPer-word coverage (examples containing the word):")
print(coverage.to_string())

print("\nGold label counts per negation word:")
print(gold_counts_df.to_string())

print("\nPredicted label counts per negation word:")
print(pred_counts_df.to_string())

print("\nGold label distribution per negation word (share within word):")
print(gold_dist_df.to_string(float_format=lambda x: f"{x:.2f}"))

print("\nPredicted label distribution per negation word (share within word):")
print(pred_dist_df.to_string(float_format=lambda x: f"{x:.2f}"))

print("\nAccuracy when a negation word is present:")
print(acc_by_word.to_string(float_format=lambda x: f"{x:.2%}"))

# Plots
def plot_distribution(dist_df, title, filename):
    ax = dist_df[labels].plot(kind="bar", stacked=True, figsize=(10, 5))
    ax.set_ylabel("Share within word")
    ax.set_xlabel("Negation word")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path = plots_dir / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


gold_plot_path = plot_distribution(
    gold_dist_df, "Gold label distribution by negation word", "negation_word_gold_distribution.png"
)
pred_plot_path = plot_distribution(
    pred_dist_df, "Predicted label distribution by negation word", "negation_word_pred_distribution.png"
)

fig, ax = plt.subplots(figsize=(8, 4))
acc_by_word.plot(kind="bar", color="seagreen", ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Negation word")
ax.set_title("Prediction accuracy when negation word is present")
plt.tight_layout()
acc_plot_path = plots_dir / "negation_word_accuracy.png"
plt.savefig(acc_plot_path, dpi=200)
plt.close()

print("\nPlots saved:")
print(f"- {gold_plot_path}")
print(f"- {pred_plot_path}")
print(f"- {acc_plot_path}")
