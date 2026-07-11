import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
RESULTS_DIR = Path("results")
DEFAULT_C = 10.0


def configure_matplotlib():
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]


def load_xy(path):
    data = np.loadtxt(path)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return x, y


def load_numeric_feature_names(path, fallback_count=14):
    names = []
    in_numeric_section = False

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "Raw numeric features:":
                in_numeric_section = True
                continue
            if line == "Raw categorical features:":
                break
            if in_numeric_section and line.startswith("- "):
                names.append(line[2:])

    if len(names) < fallback_count:
        names = [f"feature_{i + 1}" for i in range(fallback_count)]
    return names[:fallback_count]


def tune_linear_svm(x_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=LinearSVC(random_state=RANDOM_STATE, max_iter=50000),
        param_grid={"C": C_GRID},
        scoring="accuracy",
        cv=cv,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def fit_linear_svm(x_train, y_train, c_value):
    model = LinearSVC(C=c_value, random_state=RANDOM_STATE, max_iter=50000)
    model.fit(x_train, y_train)
    return model


def plot_roc_curve(y_test, scores, auc_value, output_path):
    fpr, tpr, _ = roc_curve(y_test, scores)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2.5, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1.4)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confusion_matrix(y_test, y_pred, output_path):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["0: failure", "1: success"],
    )

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_feature_weights(model, feature_names, output_path):
    weights = model.coef_[0][: len(feature_names)]
    order = np.argsort(np.abs(weights))[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_weights = weights[order]
    colors = np.where(sorted_weights >= 0, "#2563eb", "#dc2626")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_weights, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title("Linear SVM Weights for 14 Numeric Features")
    ax.set_xlabel("Weight")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_metrics_table(metrics, output_path):
    rows = [
        ("accuracy", metrics["accuracy"]),
        ("AP", metrics["AP"]),
        ("AUC", metrics["AUC"]),
        ("F1", metrics["F1"]),
        ("precision", metrics["precision"]),
        ("recall", metrics["recall"]),
        ("specificity", metrics["specificity"]),
    ]
    cell_text = [[name, f"{value:.4f}"] for name, value in rows]

    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=160)
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f9fafb" if row % 2 else "white")

    ax.set_title("Test Metrics Summary", pad=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c",
        type=float,
        default=DEFAULT_C,
        help="LinearSVC C value. Default matches the current experiment result.",
    )
    parser.add_argument(
        "--tune-c",
        action="store_true",
        help="Tune C using train-set cross-validation before plotting.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    RESULTS_DIR.mkdir(exist_ok=True)

    x_train, y_train = load_xy("train.txt")
    x_test, y_test = load_xy("test.txt")
    numeric_feature_names = load_numeric_feature_names("selected_features.txt")

    if args.tune_c:
        model, best_params, cv_accuracy = tune_linear_svm(x_train, y_train)
        selected_c = best_params["C"]
    else:
        selected_c = args.c
        model = fit_linear_svm(x_train, y_train, selected_c)
        cv_accuracy = None

    y_pred = model.predict(x_test)
    scores = model.decision_function(x_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "AP": average_precision_score(y_test, scores),
        "AUC": roc_auc_score(y_test, scores),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "specificity": specificity,
    }

    plot_roc_curve(y_test, scores, metrics["AUC"], RESULTS_DIR / "roc_curve.png")
    plot_confusion_matrix(y_test, y_pred, RESULTS_DIR / "confusion_matrix.png")
    plot_feature_weights(model, numeric_feature_names, RESULTS_DIR / "feature_weights.png")
    plot_metrics_table(metrics, RESULTS_DIR / "metrics_table.png")

    print("Saved plots to results/")
    print(f"LinearSVC C: {selected_c}")
    if cv_accuracy is not None:
        print(f"Train CV accuracy: {cv_accuracy:.6f}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
