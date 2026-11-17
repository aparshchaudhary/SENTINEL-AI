"""
SENTINEL AI – Model Evaluation & Visualization Suite
Academic Project: CSE276 – Artificial Intelligence Foundations

Generates professional publication-grade evaluation charts:
- Accuracy comparison
- Macro F1-score comparison
- Confusion matrix heatmaps for each model
- Per-class Precision vs. Recall comparison
Saves all charts into backend/ml/reports/
"""

import json
import os
import sys

# Use non-interactive backend for server/script plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class ModelEvaluator:
    """
    Renders comparative visual benchmarks and exports publication-ready graphics.
    """

    def __init__(self):
        self.reports_dir = os.path.join(BASE_DIR, "ml", "reports")
        self.metrics_path = os.path.join(self.reports_dir, "evaluation_metrics.json")
        os.makedirs(self.reports_dir, exist_ok=True)

        if not os.path.exists(self.metrics_path):
            raise FileNotFoundError(f"Metrics summary not found at {self.metrics_path}. Run train_models.py first.")

        with open(self.metrics_path, "r", encoding="utf-8") as f:
            self.metrics = json.load(f)

        self.lr_metrics = self.metrics["logistic_regression"]
        self.nb_metrics = self.metrics["multinomial_nb"]
        self.classes = ["Negative", "Neutral", "Positive"]

    def plot_accuracy_comparison(self):
        """Generates and saves the model accuracy comparison bar chart."""
        fig, ax = plt.subplots(figsize=(7, 5))
        models = ["Logistic Regression\n(L2 Regularized)", "Multinomial\nNaive Bayes"]
        accuracies = [self.lr_metrics["accuracy"] * 100, self.nb_metrics["accuracy"] * 100]
        colors = ["#2563eb", "#059669"]

        bars = ax.bar(models, accuracies, color=colors, width=0.5, edgecolor="#1e293b", linewidth=1.2)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
        ax.set_title("SENTINEL AI – Test Set Accuracy Comparison", fontsize=14, fontweight="bold", pad=15)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

        plt.tight_layout()
        out_path = os.path.join(self.reports_dir, "accuracy_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")

    def plot_macro_f1_comparison(self):
        """Generates and saves the Macro F1 and Weighted F1 comparison chart."""
        fig, ax = plt.subplots(figsize=(8, 5))
        models = ["Logistic Regression", "Multinomial Naive Bayes"]
        x = np.arange(len(models))
        width = 0.35

        macro_f1s = [self.lr_metrics["f1_macro"] * 100, self.nb_metrics["f1_macro"] * 100]
        weighted_f1s = [self.lr_metrics["f1_weighted"] * 100, self.nb_metrics["f1_weighted"] * 100]

        rects1 = ax.bar(x - width / 2, macro_f1s, width, label="Macro F1-Score", color="#4f46e5", edgecolor="#1e293b")
        rects2 = ax.bar(x + width / 2, weighted_f1s, width, label="Weighted F1-Score", color="#06b6d4", edgecolor="#1e293b")

        ax.set_ylabel("F1-Score (%)", fontsize=12, fontweight="bold")
        ax.set_title("SENTINEL AI – Macro vs. Weighted F1 Comparison", fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend(loc="lower right", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for rects in [rects1, rects2]:
            for bar in rects:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        plt.tight_layout()
        out_path = os.path.join(self.reports_dir, "macro_f1_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")

    def plot_confusion_matrix(self, model_key: str, display_name: str, filename: str):
        """Generates and saves a styled 3x3 confusion matrix heatmap."""
        cm = np.array(self.metrics[model_key]["confusion_matrix"])

        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

        fig.colorbar(cax)
        ax.set_xticks(range(len(self.classes)))
        ax.set_yticks(range(len(self.classes)))
        ax.set_xticklabels(self.classes, fontsize=11, fontweight="bold")
        ax.set_yticklabels(self.classes, fontsize=11, fontweight="bold")

        ax.set_xlabel("Predicted Polarity", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_ylabel("Ground Truth Polarity", fontsize=12, fontweight="bold")
        ax.set_title(f"Confusion Matrix: {display_name}", fontsize=13, fontweight="bold", pad=20)

        # Print counts and percentages in each cell
        total_samples = np.sum(cm)
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                count = cm[i, j]
                pct = (count / np.sum(cm[i, :])) * 100
                color = "white" if count > np.max(cm) / 2 else "black"
                ax.text(
                    j,
                    i,
                    f"{count:,}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=11,
                    fontweight="bold",
                )

        plt.tight_layout()
        out_path = os.path.join(self.reports_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")

    def plot_precision_recall_comparison(self):
        """Generates and saves class-wise precision and recall comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        x = np.arange(len(self.classes))
        width = 0.35

        for ax_idx, (model_key, display_name) in enumerate(
            [("logistic_regression", "Logistic Regression"), ("multinomial_nb", "Multinomial Naive Bayes")]
        ):
            ax = axes[ax_idx]
            model_cm = self.metrics[model_key]["class_metrics"]

            precisions = [model_cm[cls.lower()]["precision"] * 100 for cls in self.classes]
            recalls = [model_cm[cls.lower()]["recall"] * 100 for cls in self.classes]

            r1 = ax.bar(x - width / 2, precisions, width, label="Precision", color="#3b82f6", edgecolor="#1e293b")
            r2 = ax.bar(x + width / 2, recalls, width, label="Recall", color="#10b981", edgecolor="#1e293b")

            ax.set_title(display_name, fontsize=13, fontweight="bold", pad=12)
            ax.set_xticks(x)
            ax.set_xticklabels(self.classes, fontsize=11, fontweight="bold")
            ax.set_ylim(0, 100)
            ax.set_xlabel("Sentiment Class", fontsize=11, fontweight="bold")
            ax.legend(loc="upper right")
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            if ax_idx == 0:
                ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")

            for rects in [r1, r2]:
                for bar in rects:
                    h = bar.get_height()
                    ax.annotate(
                        f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

        fig.suptitle("SENTINEL AI – Class-wise Precision & Recall Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(self.reports_dir, "precision_recall_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")

    def generate_all(self):
        print("Generating visual evaluation reports...")
        self.plot_accuracy_comparison()
        self.plot_macro_f1_comparison()
        self.plot_confusion_matrix(
            "logistic_regression", "Logistic Regression", "confusion_matrix_logistic_regression.png"
        )
        self.plot_confusion_matrix(
            "multinomial_nb", "Multinomial Naive Bayes", "confusion_matrix_naive_bayes.png"
        )
        self.plot_precision_recall_comparison()
        print("All 5 evaluation visualizations generated successfully!")


def main():
    evaluator = ModelEvaluator()
    evaluator.generate_all()


if __name__ == "__main__":
    main()
