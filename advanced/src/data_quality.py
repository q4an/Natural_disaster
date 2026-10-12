"""Auditable data policy for experiments; raw CSV files are never modified."""
from pathlib import Path
import hashlib
import re
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
POLICY_VERSION = "2.0"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def message_key(text):
    """Group cosmetic variants, preserving punctuation and negation."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def parse_labels(series):
    records = []
    expected = None
    for value in series:
        pairs = [part.rsplit("-", 1) for part in str(value).split(";")]
        names = [pair[0] for pair in pairs]
        if len(names) != len(set(names)) or (expected is not None and names != expected):
            raise ValueError("Kategori şeması tutarsız veya yinelenen kategori var.")
        expected = names
        record = {name: int(number) for name, number in pairs}
        if any(v not in ({0, 1, 2} if k == "related" else {0, 1}) for k, v in record.items()):
            raise ValueError("Beklenmeyen etiket değeri.")
        records.append(record)
    return pd.DataFrame(records, index=series.index, dtype=int)


def prepare_data(messages, categories):
    """Quarantine ambiguous annotations; collapse consistent message duplicates."""
    audit = {"policy_version": POLICY_VERSION, "raw_messages": len(messages), "raw_categories": len(categories)}
    messages, categories = messages.copy(), categories.copy()
    audit["exact_message_duplicates"] = int(messages.duplicated().sum())
    audit["exact_category_duplicates"] = int(categories.duplicated().sum())
    audit["raw_missing"] = messages.isna().sum().astype(int).to_dict()
    messages = messages.drop_duplicates()
    categories = categories.drop_duplicates()
    quarantine = []

    def reject(frame, mask, reason):
        bad = frame.loc[mask].copy()
        if len(bad):
            bad["reason"] = reason
            quarantine.append(bad)
        return frame.loc[~mask].copy()

    conflict_ids = set(messages.loc[messages.id.duplicated(False), "id"]) | set(categories.loc[categories.id.duplicated(False), "id"])
    audit["conflicting_ids"] = len(conflict_ids)
    messages = reject(messages, messages.id.isin(conflict_ids), "conflicting_id_message")
    categories = reject(categories, categories.id.isin(conflict_ids), "conflicting_id_labels")
    audit["unmatched_message_ids"] = int((~messages.id.isin(categories.id)).sum())
    audit["unmatched_category_ids"] = int((~categories.id.isin(messages.id)).sum())
    messages = reject(messages, ~messages.id.isin(categories.id), "missing_categories")
    categories = reject(categories, ~categories.id.isin(messages.id), "missing_message")
    df = messages.merge(categories, on="id", validate="one_to_one")
    labels = parse_labels(df.categories)
    label_names = list(labels.columns)
    df = pd.concat([df.drop(columns="categories"), labels], axis=1)
    audit["related_2_rows"] = int(df.related.eq(2).sum())
    df = reject(df, df.related.eq(2), "related_2_meaning_unverified")
    empty = df.message.isna() | df.message.fillna("").map(message_key).eq("")
    audit["empty_messages"] = int(empty.sum())
    df = reject(df, empty, "empty_message")
    df["message_group"] = df.message.map(message_key)
    conflicting_groups = df.groupby("message_group")[label_names].nunique().max(axis=1).gt(1)
    bad_groups = set(conflicting_groups[conflicting_groups].index)
    audit["conflicting_text_groups"] = len(bad_groups)
    df = reject(df, df.message_group.isin(bad_groups), "conflicting_text_labels")
    audit["consistent_text_duplicates"] = int(df.message_group.duplicated().sum())
    # Stable representative: smallest source id, without consulting model scores.
    df = df.sort_values("id").drop_duplicates("message_group").reset_index(drop=True)
    audit["clean_rows"] = len(df)
    audit["constant_labels"] = [c for c in label_names if df[c].nunique() < 2]
    audit["quarantine_rows"] = sum(len(q) for q in quarantine)
    rejected = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=["id", "reason"])
    return df, label_names, audit, rejected


def load_curated(data_dir=None):
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    paths = {"messages": data_dir / "disaster_messages.csv", "categories": data_dir / "disaster_categories.csv"}
    result = prepare_data(pd.read_csv(paths["messages"]), pd.read_csv(paths["categories"]))
    result[2]["source_sha256"] = {k: file_hash(v) for k, v in paths.items()}
    return result
