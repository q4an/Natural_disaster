from procces_data import load_data, transform_data, cleaning_data
from train_data import split_data
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold, cross_validate

def main():
    categories, messages = load_data()
    df = transform_data(categories, messages)
    df = cleaning_data(df)

    X_train, X_test, Y_train, Y_test, category_names = split_data(df)
    label_distribution = pd.DataFrame({
        "train_pozitif": Y_train.sum(),
        "test_pozitif": Y_test.sum(),
        "train_yuzde": Y_train.mean() * 100,
        "test_yuzde": Y_test.mean() * 100,
    })
    cv = GroupKFold(n_splits=3)

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, Y_train, groups=X_train),
        start=1,
    ):
        train_messages = X_train[train_idx]
        val_messages = X_train[val_idx]
        common = set(train_messages) & set(val_messages)

        print(f"{fold}. bölme ortak mesaj sayısı: {len(common)}")

    # Baseline yalnızca eğitim verisinin CV bölmelerinde değerlendirilir.
    baseline_scores = cross_validate(
        DummyClassifier(strategy="most_frequent"),
        X_train,
        Y_train,
        groups=X_train,
        cv=cv,
        scoring={"micro_f1": "f1_micro", "macro_f1": "f1_macro"},
        error_score="raise",
    )

    baseline_results = pd.DataFrame({
        "bolme": [1, 2, 3],
        "micro_f1": baseline_scores["test_micro_f1"],
        "macro_f1": baseline_scores["test_macro_f1"],
    })
    print("\nDummyClassifier baseline sonuçları:")
    print(baseline_results.round(4).to_string(index=False))
    print(f"Ortalama micro-F1: {baseline_scores['test_micro_f1'].mean():.4f}")
    print(f"Ortalama macro-F1: {baseline_scores['test_macro_f1'].mean():.4f}")

    print(label_distribution.round(2).to_string())

    common_messages = set(X_train).intersection(set(X_test))

    print(f"Eğitim satır sayısı: {len(X_train)}")
    print(f"Test satır sayısı: {len(X_test)}")
    print(f"Ortak mesaj sayısı: {len(common_messages)}")

    if common_messages:
        print("Aynı mesajlar hem eğitim hem test kümesinde bulunuyor.")
    else:
        print("Eğitim ve test kümelerinde ortak mesaj yok.")


if __name__ == "__main__":
    main()




