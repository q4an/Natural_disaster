from pathlib import Path

import joblib

from text_processing import tokenize


bundle_path=Path(__file__).resolve().parent.parent / "models" / "disaster_model_bundle.pkl"








if bundle_path.is_file():
    bundle_model = joblib.load(bundle_path)
    model = bundle_model["model"]
    category_names = bundle_model["category_names"]

    print("Model ve kategori isimleri yüklendi")

    message = "We need food and drinking water "
    predict = model.predict([message])

    print("Mesaj :",message)
    print(f"Tahmin :",predict)

    if len(category_names) != len(predict[0]):
        raise ValueError(
            "Kategori sayısı ile model çıktısı uyuşmuyor."
        )

    selected_categories = []

    for category, value in zip(
        category_names,
        predict[0]
    ):
        if value == 1:
            selected_categories.append(category)

    print(
        "Kategoriler:",
        ", ".join(selected_categories)
        or "Kategori seçilmedi"
    )

else:
    print(
        f"Model paketi bulunamadı: {bundle_path}"
    )