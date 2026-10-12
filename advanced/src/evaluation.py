"""Metrics and reports shared by training and the local demo."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report,
                             f1_score, hamming_loss, precision_score, recall_score)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metrics(y, pred):
    return {
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y, pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y, pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y, pred)),
        "subset_accuracy": float(np.mean(np.all(np.asarray(y) == pred, axis=1))),
    }


def probabilities(model, x):
    scores = model.predict_proba(x)
    if isinstance(scores, list):
        heads = model.named_steps["clf"].estimators_ if hasattr(model, "named_steps") else None
        classes = [head.classes_ for head in heads] if heads else model.classes_
        scores = np.column_stack([
            p[:, list(c).index(1)] if 1 in c else np.zeros(len(x))
            for p, c in zip(scores, classes)
        ])
    return np.asarray(scores)


def choose_thresholds(y, scores, names, min_positives=20):
    """Validation-only per-label F1 selection; conservative tie and support rule."""
    rows = []
    for i, name in enumerate(names):
        truth = np.asarray(y)[:, i]
        threshold = 0.5
        eligible = int(truth.sum()) >= min_positives and int((1 - truth).sum()) >= min_positives
        if eligible:
            grid = np.arange(0.05, 0.951, 0.05)
            values = [f1_score(truth, scores[:, i] >= t, zero_division=0) for t in grid]
            best = max(values)
            tied = [t for t, v in zip(grid, values) if np.isclose(v, best)]
            threshold = float(min(tied, key=lambda t: (abs(t - 0.5), -t)))
        rows.append({"label": name, "threshold": threshold, "positive_support": int(truth.sum()), "tuned": eligible})
    return pd.DataFrame(rows)


def per_label(y, pred, scores, names):
    truth = np.asarray(y)
    report = classification_report(truth, pred, target_names=names, output_dict=True, zero_division=0)
    rows = []
    for i, name in enumerate(names):
        t, p = truth[:, i], pred[:, i]
        row = {"label": name, **report[name]}
        row.update(tp=int(((t == 1) & (p == 1)).sum()), fp=int(((t == 0) & (p == 1)).sum()),
                   fn=int(((t == 1) & (p == 0)).sum()), tn=int(((t == 0) & (p == 0)).sum()))
        row["average_precision"] = float(average_precision_score(t, scores[:, i])) if t.sum() else None
        row["brier"] = float(np.mean((scores[:, i] - t) ** 2))
        rows.append(row)
    return pd.DataFrame(rows)


def error_analysis(frame, y, pred, scores, names, out):
    """Save validation errors, not test examples used for iterative development."""
    truth = np.asarray(y)
    rows = []
    for i, label in enumerate(names):
        for j in np.flatnonzero(truth[:, i] != pred[:, i]):
            row = frame.iloc[j]
            rows.append({"id": int(row.id), "label": label, "error": "FN" if truth[j, i] else "FP",
                         "score": float(scores[j, i]), "genre": row.genre,
                         "length": len(row.message), "message": row.message})
    pd.DataFrame(rows, columns=["id", "label", "error", "score", "genre", "length", "message"]).to_csv(out / "validation_errors.csv", index=False)
    segment_rows = []
    segments = {"genre": frame.genre.fillna("unknown").to_numpy(),
                "length": pd.cut(frame.message.str.len(), [0, 80, 200, 500, np.inf],
                                 labels=["1-80", "81-200", "201-500", "501+"]).astype(str).to_numpy()}
    for kind, values in segments.items():
        for value in np.unique(values):
            mask = values == value
            segment_rows.append({"segment": kind, "value": value, "rows": int(mask.sum()), **metrics(truth[mask], pred[mask])})
    pd.DataFrame(segment_rows).to_csv(out / "validation_segments.csv", index=False)


def eda(frame, names, out):
    counts = frame[names].sum().sort_values()
    pd.DataFrame({"positive": counts, "rate": counts / len(frame)}).to_csv(out / "development_label_distribution.csv", index_label="label")
    fig, axes = plt.subplots(1, 3, figsize=(17, 10), gridspec_kw={"width_ratios": [2, 1, 1]})
    counts.plot.barh(ax=axes[0], color="#168b89")
    axes[0].set_title("Development: positive labels")
    frame.genre.value_counts().plot.bar(ax=axes[1], color="#3157a6", rot=30)
    axes[1].set_title("Message source")
    axes[2].hist(frame.message.str.len(), bins=np.geomspace(1, max(2, frame.message.str.len().max()), 40), color="#d68b29")
    axes[2].set_xscale("log")
    axes[2].set_title("Length (characters, log axis)")
    fig.tight_layout()
    fig.savefig(out / "eda.png", dpi=150)
    plt.close(fig)
    cooccurrence = frame[names].T.dot(frame[names])
    cooccurrence.to_csv(out / "label_cooccurrence.csv")


def reliability(y, scores, names, out):
    rows = []
    focus = [n for n in ["related", "water", "food", "medical_help", "search_and_rescue"] if n in names]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    for name in focus:
        i = names.index(name)
        x, observed = [], []
        bins = np.minimum((scores[:, i] * 10).astype(int), 9)
        for b in range(10):
            mask = bins == b
            if not mask.any():
                continue
            mean = float(scores[mask, i].mean())
            rate = float(np.asarray(y)[mask, i].mean())
            rows.append({"label": name, "bin": b, "rows": int(mask.sum()), "mean_score": mean, "positive_rate": rate})
            x.append(mean)
            observed.append(rate)
        ax.plot(x, observed, marker="o", label=name)
    ax.set(xlabel="Mean model score", ylabel="Observed positive rate", title="Validation reliability (diagnostic only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "reliability_bins.csv", index=False)


def explain_global(model, names, out):
    features = model.named_steps["tfidf"].get_feature_names_out()
    rows = []
    for label, head in zip(names, model.named_steps["clf"].estimators_):
        signed = hasattr(head, "coef_")
        weights = head.coef_[0] if signed else head.feature_importances_
        positions = np.argsort(weights)
        for direction, chosen in [("positive" if signed else "importance", positions[-15:][::-1])]:
            for j in chosen:
                rows.append({"label": label, "feature": features[j], "weight": float(weights[j]), "direction": direction})
        if signed:
            for j in positions[:15]:
                rows.append({"label": label, "feature": features[j], "weight": float(weights[j]), "direction": "negative"})
    pd.DataFrame(rows).to_csv(out / "feature_effects.csv", index=False)
