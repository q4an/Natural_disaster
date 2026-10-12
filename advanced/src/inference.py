"""Load the versioned experiment artifact and apply its frozen thresholds."""
import json
from pathlib import Path

import joblib
import numpy as np

from data_quality import ROOT, file_hash
from evaluation import probabilities


def load_latest():
    pointer = ROOT / "models" / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError("Önce python src/experiment.py komutunu çalıştırın.")
    manifest = json.loads(pointer.read_text(encoding="utf-8"))
    path = (ROOT / manifest["model"]).resolve()
    if not path.is_relative_to((ROOT / "models").resolve()):
        raise ValueError("Model yolu proje models klasöründe olmalı.")
    if file_hash(path) != manifest["model_sha256"]:
        raise ValueError("Model dosyasının SHA256 özeti manifest ile uyuşmuyor.")
    bundle = joblib.load(path)  # Only load this project's trusted local artifact.
    if bundle.get("schema_version") != 2:
        raise ValueError("Desteklenmeyen model paketi sürümü.")
    if len(bundle["category_names"]) != len(bundle["thresholds"]):
        raise ValueError("Kategori ve eşik sayısı uyuşmuyor.")
    return bundle


def predict_message(bundle, message):
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Lütfen boş olmayan bir İngilizce mesaj girin.")
    if len(message) > 5000:
        raise ValueError("Mesaj en fazla 5000 karakter olabilir.")
    message = message.strip()
    pipeline = bundle["model"]
    features = pipeline.named_steps["tfidf"].transform([message])
    metadata = bundle["metadata"]
    result = {"run_id": metadata["run_id"], "model": metadata["selected_model"],
              "language": "English", "calibrated": False,
              "excluded_labels": metadata["excluded_labels"], "categories": [], "scores": [],
              "review_note": "Eğitim demosu: skorlar kalibre edilmiş olasılık değildir; yönlendirmeyi insan doğrulamalı."}
    if features.nnz == 0:
        result["review_note"] = "Modelin sözlüğünde eşleşen sözcük yok. İngilizce bir mesaj girin; bu sonuçla yönlendirme yapmayın."
        return result
    values = probabilities(pipeline, [message])[0]
    if len(values) != len(bundle["category_names"]) or not np.isfinite(values).all():
        raise ValueError("Model çıktısı kategori şemasıyla uyumsuz.")
    names = pipeline.named_steps["tfidf"].get_feature_names_out()
    heads = pipeline.named_steps["clf"].estimators_
    for i, (category, threshold, value) in enumerate(zip(bundle["category_names"], bundle["thresholds"], values)):
        selected = bool(value >= threshold)
        item = {"category": category, "score": float(value), "threshold": float(threshold), "selected": selected, "evidence": []}
        head = heads[i]
        if hasattr(head, "coef_"):
            contributions = features.data * head.coef_[0][features.indices]
            for pos in np.argsort(contributions)[-4:][::-1]:
                if contributions[pos] > 0:
                    item["evidence"].append({"term": names[features.indices[pos]], "contribution": float(contributions[pos])})
        if selected:
            result["categories"].append(category)
        result["scores"].append(item)
    result["scores"].sort(key=lambda row: row["score"], reverse=True)
    return result
