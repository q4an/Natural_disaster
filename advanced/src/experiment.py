"""Reproducible, group-aware model comparison and one final test evaluation.

Run from the project root: python src/experiment.py
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import logging
from pathlib import Path
import platform
import shutil
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from data_quality import ROOT, POLICY_VERSION, file_hash, load_curated
from evaluation import (choose_thresholds, eda, error_analysis, explain_global,
                        metrics, per_label, probabilities, reliability, write_json)

SEED = 42
logger = logging.getLogger(__name__)


def split_frames(df):
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(df, groups=df.message_group))
    pool = df.iloc[train]
    development, validation = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(pool, groups=pool.message_group))
    frames = {"development": pool.iloc[development].reset_index(drop=True),
              "validation": pool.iloc[validation].reset_index(drop=True),
              "test": df.iloc[test].reset_index(drop=True)}
    keys = list(frames)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            if set(frames[left].message_group) & set(frames[right].message_group):
                raise ValueError("Bölmeler arasında grup sızıntısı bulundu.")
    return frames


def candidates(include_xgb=True):
    def text_pipeline(classifier):
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True)),
            ("clf", classifier),
        ])
    models = {"dummy": DummyClassifier(strategy="most_frequent")}
    for c in [1.0, 4.0]:
        models[f"logistic_c{int(c)}"] = text_pipeline(OneVsRestClassifier(
            LogisticRegression(C=c, solver="liblinear", max_iter=500, random_state=SEED), n_jobs=1))
    if include_xgb:
        from xgboost import XGBClassifier
        models["xgboost_depth4"] = text_pipeline(MultiOutputClassifier(XGBClassifier(
            n_estimators=60, max_depth=4, learning_rate=0.1, colsample_bytree=0.75,
            tree_method="hist", max_bin=64, n_jobs=2, random_state=SEED,
            objective="binary:logistic", eval_metric="logloss"), n_jobs=1))
    return models


def learning_curve(model, frame, names, split, out):
    train_idx, val_idx = split
    training, validation = frame.iloc[train_idx], frame.iloc[val_idx]
    rows = []
    for fraction in [0.25, 0.5, 1.0]:
        if fraction < 1:
            subset, _ = next(GroupShuffleSplit(n_splits=1, train_size=fraction, random_state=SEED).split(training, groups=training.message_group))
            part = training.iloc[subset]
        else:
            part = training
        if (part[names].nunique() < 2).any():
            rows.append({"fraction": fraction, "rows": len(part), "status": "skipped_single_class"})
            continue
        logger.info("Learning curve: %.0f%% (%s rows)", fraction * 100, len(part))
        fitted = clone(model).fit(part.message.to_numpy(), part[names])
        row = {"fraction": fraction, "rows": len(part), "status": "ok"}
        for label, data in [("train", part), ("validation", validation)]:
            result = metrics(data[names], fitted.predict(data.message.to_numpy()))
            row.update({f"{label}_{key}": value for key, value in result.items() if key.endswith("f1")})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "learning_curve.csv", index=False)
    good = table[table.status == "ok"]
    if len(good):
        fig, ax = plt.subplots(figsize=(7, 5))
        for col in ["train_micro_f1", "validation_micro_f1", "train_macro_f1", "validation_macro_f1"]:
            ax.plot(good.rows, good[col], "o-", label=col)
        ax.set(xlabel="Training rows", ylabel="F1", title="Learning curve: one fixed development fold, threshold 0.5")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "learning_curve.png", dpi=150)
        plt.close(fig)


def run(out, include_xgb):
    started = time.perf_counter()
    df, all_labels, audit, rejected = load_curated()
    write_json(out / "data_audit.json", audit)
    rejected.to_csv(out / "quarantine.csv", index=False)
    frames = split_frames(df)
    dev, val, test = (frames[k] for k in ["development", "validation", "test"])
    # Target availability is decided using development data, never test outcomes.
    excluded = [name for name in all_labels if dev[name].nunique() < 2]
    names = [name for name in all_labels if name not in excluded]
    if not names:
        raise ValueError("Öğrenilebilir hedef yok.")
    eda(dev, all_labels, out)
    pd.concat([frame[["id", "message_group"]].assign(split=split) for split, frame in frames.items()]).to_csv(out / "split_manifest.csv", index=False)
    write_json(out / "split_summary.json", {"rows": {k: len(v) for k, v in frames.items()}, "pairwise_group_overlap": 0,
                                            "labels": names, "excluded_constant_development_labels": excluded})
    x, y, groups = dev.message.to_numpy(), dev[names], dev.message_group.to_numpy()
    splits = list(GroupKFold(n_splits=3).split(x, y, groups=groups))
    fold_rows, support_rows = [], []
    for fold, (tr, va) in enumerate(splits, 1):
        if set(groups[tr]) & set(groups[va]):
            raise ValueError("CV grup sızıntısı.")
        if (y.iloc[tr].nunique() < 2).any():
            raise ValueError("CV eğitim katında tek sınıflı hedef; bölme politikasını inceleyin.")
        fold_rows.append({"fold": fold, "train_rows": len(tr), "validation_rows": len(va), "group_overlap": 0})
        for name in names:
            support_rows.append({"fold": fold, "label": name, "train_positive": int(y.iloc[tr][name].sum()),
                                 "validation_positive": int(y.iloc[va][name].sum())})
    pd.DataFrame(fold_rows).to_csv(out / "cv_group_checks.csv", index=False)
    pd.DataFrame(support_rows).to_csv(out / "cv_label_support.csv", index=False)
    models = candidates(include_xgb)
    hypotheses = {
        "dummy": "Majority labels establish a no-text baseline.",
        "logistic_c1": "Sparse lexical evidence can separate labels with a regularized linear model.",
        "logistic_c4": "Weaker regularization can improve recall without an excessive train-validation gap.",
        "xgboost_depth4": "Shallow boosted trees can improve on additive lexical evidence through interactions.",
    }
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    for path in (ROOT / "src").glob("*.py"):
        shutil.copy2(path, source_dir / path.name)
    config = {"run_id": out.name, "created_utc": datetime.now(timezone.utc).isoformat(),
              "seed": SEED, "data_policy": POLICY_VERSION, "source_sha256": audit["source_sha256"],
              "code_sha256": {p.name: file_hash(p) for p in source_dir.glob("*.py")},
              "python": platform.python_version(), "packages": {},
              "candidates": {k: repr(v) for k, v in models.items()}, "hypotheses": hypotheses,
              "selection_metric": "mean development CV micro-F1 at threshold 0.5",
              "threshold_policy": "validation per-label F1; grid 0.05..0.95; >=20 positives and negatives; ties nearest 0.5",
              "calibration": "not fitted; validation Brier and reliability diagnostics only",
              "split_rows": {k: len(v) for k, v in frames.items()}, "excluded_labels": excluded}
    for package in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib", "matplotlib"]:
        try:
            config["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(out / "run_config.json", config)
    cv_rows = []
    for name, candidate in models.items():
        for fold, (tr, va) in enumerate(splits, 1):
            logger.info("CV %s, fold %s/3", name, fold)
            tick = time.perf_counter()
            fitted = clone(candidate).fit(x[tr], y.iloc[tr])
            row = {"model": name, "fold": fold, "fit_seconds": time.perf_counter() - tick}
            for prefix, indices in [("train", tr), ("validation", va)]:
                row.update({f"{prefix}_{key}": value for key, value in metrics(y.iloc[indices], fitted.predict(x[indices])).items()})
            cv_rows.append(row)
            pd.DataFrame(cv_rows).to_csv(out / "cv_results.csv", index=False)
            logger.info("%s fold %s micro-F1=%.4f macro-F1=%.4f", name, fold, row["validation_micro_f1"], row["validation_macro_f1"])
    results = pd.DataFrame(cv_rows)
    summary = results.groupby("model").agg(
        micro_f1=("validation_micro_f1", "mean"), micro_f1_std=("validation_micro_f1", "std"),
        macro_f1=("validation_macro_f1", "mean"), train_micro_f1=("train_micro_f1", "mean"),
        mean_fit_seconds=("fit_seconds", "mean"))
    summary["train_validation_gap"] = summary.train_micro_f1 - summary.micro_f1
    summary.to_csv(out / "model_comparison.csv")
    winner = summary.micro_f1.idxmax()
    if winner == "dummy":
        raise RuntimeError("Hiçbir metin modeli baseline'ı geçmedi; demo modeli yayımlanmadı. CV raporlarını inceleyin.")
    logger.info("Selected by development CV: %s", winner)
    learning_curve(models[winner], dev, names, splits[0], out)
    fitted = clone(models[winner]).fit(x, y)
    val_scores = probabilities(fitted, val.message.to_numpy())
    threshold_table = choose_thresholds(val[names], val_scores, names)
    threshold_table.to_csv(out / "thresholds.csv", index=False)
    thresholds = threshold_table.threshold.to_numpy()
    val_pred = (val_scores >= thresholds).astype(int)
    validation = {"default_0_5": metrics(val[names], (val_scores >= 0.5).astype(int)), "tuned": metrics(val[names], val_pred),
                  "note": "Thresholds were selected on this validation set; tuned scores are optimistic selection results."}
    write_json(out / "validation_metrics.json", validation)
    per_label(val[names], val_pred, val_scores, names).to_csv(out / "validation_per_label.csv", index=False)
    error_analysis(val, val[names], val_pred, val_scores, names, out)
    reliability(val[names], val_scores, names, out)
    explain_global(fitted, names, out)
    # Freeze model + thresholds BEFORE inspecting test metrics. No refit on validation:
    # doing so would change the score distribution on which thresholds were chosen.
    model_dir = ROOT / "models" / out.name
    model_dir.mkdir(exist_ok=False)
    bundle = {"schema_version": 2, "model": fitted, "category_names": names, "thresholds": thresholds,
              "metadata": {**config, "selected_model": winner, "language": "English",
                           "excluded_labels": excluded, "calibrated": False}}
    bundle_path = model_dir / "bundle.joblib"
    temporary = model_dir / "bundle.tmp"
    joblib.dump(bundle, temporary)
    temporary.replace(bundle_path)
    test_scores = probabilities(fitted, test.message.to_numpy())
    test_pred = (test_scores >= thresholds).astype(int)
    final = {"default_0_5": metrics(test[names], (test_scores >= 0.5).astype(int)), "frozen_thresholds": metrics(test[names], test_pred)}
    write_json(out / "test_metrics.json", final)
    per_label(test[names], test_pred, test_scores, names).to_csv(out / "test_per_label.csv", index=False)
    # Test distributions are published only after the model and thresholds are frozen.
    pd.DataFrame({f"{part}_{stat}": (frame[names].sum() if stat == "positive" else frame[names].mean())
                  for part, frame in frames.items() for stat in ["positive", "rate"]}).to_csv(out / "split_label_distribution.csv", index_label="label")
    manifest = {"run_id": out.name, "model": bundle_path.relative_to(ROOT).as_posix(), "report": out.relative_to(ROOT).as_posix(),
                "model_sha256": file_hash(bundle_path), "selected_model": winner, "elapsed_seconds": time.perf_counter() - started}
    write_json(out / "artifact_manifest.json", manifest)
    lines = [f"# Deney: {out.name}", "", f"Seçilen model: **{winner}** (development CV micro-F1).", "",
             "Veri politikası v2: çelişkili ve anlamı doğrulanmamış etiketler karantinada; eski skorlarla doğrudan kıyaslanmaz.", "",
             "| Model | CV micro-F1 | CV macro-F1 | Train-CV farkı |", "|---|---:|---:|---:|"]
    for name, row in summary.iterrows():
        lines.append(f"| {name} | {row.micro_f1:.4f} | {row.macro_f1:.4f} | {row.train_validation_gap:.4f} |")
    lines += ["", "## Eşik seçimi ve son test", "",
              f"Validation micro-F1: varsayılan {validation['default_0_5']['micro_f1']:.4f}, eşik seçimi sonrası {validation['tuned']['micro_f1']:.4f} (seçimde kullanılan veri).",
              f"Son test micro-F1: {final['frozen_thresholds']['micro_f1']:.4f}; macro-F1: {final['frozen_thresholds']['macro_f1']:.4f}.",
              "Test sonucu hiçbir model veya eşik kararında kullanılmadı. Bu test artık görülmüştür; sonraki geliştirmeler validation üzerinde yapılmalıdır.", "",
              "## İnceleme dosyaları", "", "- `validation_errors.csv`: etiket başına yanlış pozitif ve negatif mesajlar.",
              "- `validation_segments.csv`: kaynak ve uzunluğa göre performans.",
              "- `learning_curve.png`: tek sabit CV katında veri miktarı deneyi; kesin overfitting/underfitting teşhisi değildir.",
              "- `feature_effects.csv`: doğrusal katsayılar veya ağaç özellik önemleri; nedensel açıklama değildir.",
              "- `reliability.png`: skorların güvenilirlik incelemesi; kalibrasyon uygulanmadı.",
              "- `quarantine.csv`: elle incelenecek dışlanan kayıtlar.", "",
              "## Deney günlüğü", "", "**Değişiklik:** Veri politikası, bağımsız bölmeler, model karşılaştırması ve validation eşikleri.",
              "**Hipotez:** Sözcük bilgisi kullanan modeller çoğunluk baseline'ını geçer.",
              f"**Validation sonucu:** {winner}, CV micro-F1 {summary.loc[winner, 'micro_f1']:.4f}.",
              f"**Önceki sonuç:** Aynı veride dummy CV micro-F1 {summary.loc['dummy', 'micro_f1']:.4f}.",
              f"**Değişim:** {summary.loc[winner, 'micro_f1'] - summary.loc['dummy', 'micro_f1']:.4f} mutlak F1.",
              "**Yorum:** Eşik seçimi ayrı validation üzerinde; test yalnız son ölçümde açıldı.",
              "**Sonraki karar:** Validation yanlış negatiflerini oku; yeni özellik deneyi için hipotez yaz."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in [ROOT / "models" / "latest.json", ROOT / "reports" / "latest.json"]:
        tmp = path.with_suffix(".tmp")
        write_json(tmp, manifest)
        tmp.replace(path)
    logger.info("Complete: %s; test micro-F1 %.4f", out, final["frozen_thresholds"]["micro_f1"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-xgb", action="store_true", help="Run dummy and two linear candidates only.")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    out = ROOT / "reports" / run_id
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(out / "training.log", encoding="utf-8")])
    write_json(out / "status.json", {"status": "running"})
    try:
        run(out, not args.skip_xgb)
    except Exception as error:
        write_json(out / "status.json", {"status": "failed", "error": str(error)})
        logger.exception("Experiment failed; partial reports preserved.")
        raise
    write_json(out / "status.json", {"status": "complete"})


if __name__ == "__main__":
    main()
