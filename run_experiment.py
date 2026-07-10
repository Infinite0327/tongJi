import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
TARGET_COLUMN = "\u5bab\u5185\u598a\u5a20\u672f\u540e3-4\u5468\uff08\u80ce\u5fc3\u80ce\u82bd1\uff09"
DEFAULT_EXCEL_PATHS = [
    Path("\u75c5\u5386\u539f\u59cb\u6570\u636e.xlsx"),
    Path("\u8d85\u58f0\u6570\u636e") / "\u75c5\u5386\u539f\u59cb\u6570\u636e.xlsx",
]

LEAKAGE_KEYWORDS = [
    "id",
    "\u7f16\u53f7",
    "\u5e8f\u53f7",
    "\u75c5\u6848",
    "\u75c5\u5386",
    "\u4f4f\u9662\u53f7",
    "\u95e8\u8bca\u53f7",
    "\u59d3\u540d",
    "\u540d\u5b57",
    "name",
    "\u7535\u8bdd",
    "\u8eab\u4efd\u8bc1",
    "\u6807\u7b7e",
    "label",
    "target",
    "\u598a\u5a20",
    "\u7ed3\u5c40",
    "\u7ed3\u679c",
    "\u672f\u540e",
    "hcg",
    "\u03b2",
    "beta",
    "\u80ce\u5fc3",
    "\u80ce\u82bd",
    "\u6d41\u4ea7",
    "\u6d3b\u4ea7",
    "\u751f\u5316",
]

MISSING_MARKERS = {"", "-", "\u2014", "\u2013", "NA", "N/A", "nan", "NaN", "\u65e0"}
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
MIN_CATEGORY_COUNT = 3
MIN_NUMERIC_COVERAGE = 0.95


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=3):
        self.min_count = min_count

    def fit(self, x, y=None):
        frame = pd.DataFrame(x).astype(str)
        self.keep_values_ = []
        for column in frame.columns:
            counts = frame[column].value_counts(dropna=False)
            keep = set(counts[counts >= self.min_count].index)
            self.keep_values_.append(keep)
        return self

    def transform(self, x):
        frame = pd.DataFrame(x).astype(str).copy()
        for i, column in enumerate(frame.columns):
            keep = self.keep_values_[i]
            frame[column] = frame[column].where(frame[column].isin(keep), "__RARE__")
        return frame.to_numpy(dtype=object)


def find_excel_path(cli_path):
    if cli_path:
        path = Path(cli_path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")
        return path

    for path in DEFAULT_EXCEL_PATHS:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find the Excel file.")


def col_to_index(cell_ref):
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch.upper()) - ord("A") + 1)
    return index - 1


def xml_text(element):
    if element is None:
        return ""
    return "".join(element.itertext())


def read_xlsx_without_openpyxl(path):
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    with zipfile.ZipFile(path) as zf:
        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", ns):
                shared_strings.append(xml_text(si))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        first_sheet = workbook.find("main:sheets/main:sheet", ns)
        if first_sheet is None:
            raise ValueError("Workbook has no sheets.")

        rel_id = first_sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall("rel:Relationship", ns):
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib["Target"]
                break
        if target is None:
            raise ValueError("Could not resolve first worksheet relationship.")

        sheet_path = str(Path("xl") / target).replace("\\", "/")
        sheet_xml = ET.fromstring(zf.read(sheet_path))

        rows = []
        for row in sheet_xml.findall(".//main:sheetData/main:row", ns):
            values = []
            for cell in row.findall("main:c", ns):
                ref = cell.attrib.get("r", "")
                idx = col_to_index(ref)
                while len(values) <= idx:
                    values.append(np.nan)

                cell_type = cell.attrib.get("t")
                value_node = cell.find("main:v", ns)
                inline_node = cell.find("main:is", ns)

                if cell_type == "s":
                    raw = xml_text(value_node)
                    value = shared_strings[int(raw)] if raw != "" else np.nan
                elif cell_type == "inlineStr":
                    value = xml_text(inline_node)
                else:
                    raw = xml_text(value_node)
                    value = raw if raw != "" else np.nan
                    if isinstance(value, str):
                        try:
                            numeric = float(value)
                            value = int(numeric) if numeric.is_integer() else numeric
                        except ValueError:
                            pass
                values[idx] = value
            rows.append(values)

    while rows and all(pd.isna(v) or v == "" for v in rows[0]):
        rows.pop(0)
    if not rows:
        raise ValueError("Worksheet is empty.")

    width = max(len(row) for row in rows)
    rows = [row + [np.nan] * (width - len(row)) for row in rows]
    header = [str(v).strip() if not pd.isna(v) else f"Unnamed: {i}" for i, v in enumerate(rows[0])]
    return pd.DataFrame(rows[1:], columns=header)


def read_excel(path):
    try:
        return pd.read_excel(path, sheet_name=0)
    except ImportError as exc:
        if "openpyxl" not in str(exc).lower():
            raise
        print("openpyxl is not installed; using built-in read-only xlsx parser.")
        return read_xlsx_without_openpyxl(path)


def normalize_column_name(name):
    return re.sub(r"\s+", "", str(name)).lower()


def is_target_column(column):
    return normalize_column_name(column) == normalize_column_name(TARGET_COLUMN)


def label_to_binary(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        unique = sorted(v for v in numeric.dropna().unique())
        if set(unique).issubset({0, 1}):
            return numeric.astype(int)

    positive_tokens = [
        "1",
        "\u662f",
        "\u6709",
        "\u6210\u529f",
        "\u9633\u6027",
        "\u598a\u5a20",
        "\u6000\u5b55",
        "\u4e34\u5e8a\u598a\u5a20",
        "\u89c1\u80ce\u5fc3",
    ]
    negative_tokens = [
        "0",
        "\u5426",
        "\u65e0",
        "\u5931\u8d25",
        "\u9634\u6027",
        "\u672a\u598a\u5a20",
        "\u672a\u5b55",
        "\u672a\u89c1",
    ]

    def convert(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        lowered = text.lower()
        if lowered in {"1", "1.0", "true", "yes", "y"}:
            return 1
        if lowered in {"0", "0.0", "false", "no", "n"}:
            return 0
        if any(token in text for token in negative_tokens):
            return 0
        if any(token in text for token in positive_tokens):
            return 1
        return np.nan

    converted = series.map(convert)
    return converted.astype("Int64")


def is_leakage_or_identifier(column):
    normalized = normalize_column_name(column)
    if normalized.startswith("unnamed:"):
        return True
    if is_target_column(column):
        return True
    return any(keyword.lower() in normalized for keyword in LEAKAGE_KEYWORDS)


def replace_missing_markers(series):
    return series.replace(list(MISSING_MARKERS), np.nan)


def coerce_numeric_series(series):
    return pd.to_numeric(replace_missing_markers(series), errors="coerce")


def split_feature_columns(df):
    numeric_features = []
    categorical_features = []
    excluded = []

    for column in df.columns:
        if is_leakage_or_identifier(column):
            excluded.append((column, "identifier/label/outcome leakage keyword"))
            continue

        cleaned = replace_missing_markers(df[column])
        non_missing = int(cleaned.notna().sum())
        if non_missing == 0:
            excluded.append((column, "empty"))
            continue

        numeric = pd.to_numeric(cleaned, errors="coerce")
        numeric_non_missing = int(numeric.notna().sum())
        numeric_coverage = numeric_non_missing / len(df) if len(df) else 0.0
        if numeric_non_missing == non_missing and numeric_coverage >= MIN_NUMERIC_COVERAGE:
            numeric_features.append(column)
        elif numeric_non_missing == non_missing:
            excluded.append((column, "numeric but low coverage"))
        else:
            categorical_features.append(column)

    return numeric_features, categorical_features, excluded


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(numeric_features, categorical_features, c_value):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler(clip=True)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("rare", RareCategoryGrouper(min_count=MIN_CATEGORY_COUNT)),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("svm", LinearSVC(C=c_value, random_state=RANDOM_STATE, max_iter=50000)),
        ]
    )


def transformed_feature_names(best_model, numeric_features, categorical_features):
    names = []
    names.extend(numeric_features)

    if categorical_features:
        preprocessor = best_model.named_steps["preprocess"]
        cat_pipeline = preprocessor.named_transformers_["cat"]
        onehot = cat_pipeline.named_steps["onehot"]
        try:
            encoded = onehot.get_feature_names_out(categorical_features)
        except AttributeError:
            encoded = onehot.get_feature_names(categorical_features)
        names.extend([str(name) for name in encoded])

    return names


def save_txt(path, x_values, y_values):
    data = np.column_stack([x_values, y_values])
    np.savetxt(path, data, fmt="%.10g")


def best_threshold_for_accuracy(scores, y_true):
    order = np.argsort(scores)
    sorted_scores = scores[order]
    thresholds = [sorted_scores[0] - 1.0]
    thresholds.extend(((sorted_scores[:-1] + sorted_scores[1:]) / 2.0).tolist())
    thresholds.append(sorted_scores[-1] + 1.0)

    best_threshold = 0.0
    best_accuracy = -1.0
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def select_c_and_threshold(x_train, y_train, numeric_features, categorical_features):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for c_value in C_GRID:
        oof_scores = np.zeros(len(y_train), dtype=float)
        for fold_train_idx, fold_valid_idx in cv.split(x_train, y_train):
            fold_model = build_pipeline(numeric_features, categorical_features, c_value)
            fold_model.fit(x_train.iloc[fold_train_idx], y_train[fold_train_idx])
            oof_scores[fold_valid_idx] = fold_model.decision_function(
                x_train.iloc[fold_valid_idx]
            )

        threshold, cv_accuracy = best_threshold_for_accuracy(oof_scores, y_train)
        results.append(
            {
                "C": float(c_value),
                "threshold": float(threshold),
                "cv_accuracy": float(cv_accuracy),
            }
        )

    best = max(results, key=lambda item: (item["cv_accuracy"], -item["C"]))
    return best, results


def write_selected_features(path, numeric_features, categorical_features, model_features):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Raw numeric features:\n")
        for feature in numeric_features:
            f.write(f"- {feature}\n")

        f.write("\nRaw categorical features:\n")
        for feature in categorical_features:
            f.write(f"- {feature}\n")

        f.write("\nExpanded model vector features:\n")
        for i, feature in enumerate(model_features, 1):
            f.write(f"{i}. {feature}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", default=None, help="Path to the Excel data file")
    args = parser.parse_args()

    excel_path = find_excel_path(args.excel)
    df = read_excel(excel_path)
    df = df.dropna(how="all").reset_index(drop=True)

    print(f"Excel file: {excel_path}")
    print("All columns:")
    for i, column in enumerate(df.columns, 1):
        print(f"{i}. {column}")

    target_matches = [column for column in df.columns if is_target_column(column)]
    if not target_matches:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")
    target_column = target_matches[0]

    y = label_to_binary(df[target_column])
    usable_mask = y.notna()
    df = df.loc[usable_mask].reset_index(drop=True)
    y = y.loc[usable_mask].astype(int).to_numpy()

    numeric_features, categorical_features, excluded = split_feature_columns(df)
    if len(df) < 200:
        raise ValueError(f"Need at least 200 usable labeled rows for 160/40 split; got {len(df)}.")
    if not numeric_features and not categorical_features:
        raise ValueError("No usable non-leakage features found.")

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=160,
        test_size=40,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2 else None,
    )

    x = df[numeric_features + categorical_features].copy()
    for column in numeric_features:
        x[column] = coerce_numeric_series(x[column])
    for column in categorical_features:
        x[column] = replace_missing_markers(x[column]).astype(object)

    x_train = x.iloc[train_idx].reset_index(drop=True)
    x_test = x.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    best_search, cv_results = select_c_and_threshold(
        x_train,
        y_train,
        numeric_features,
        categorical_features,
    )
    best_model = build_pipeline(
        numeric_features,
        categorical_features,
        best_search["C"],
    )
    best_model.fit(x_train, y_train)

    scores = best_model.decision_function(x_test)
    y_pred = (scores >= best_search["threshold"]).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred)

    x_train_vector = best_model.named_steps["preprocess"].transform(x_train)
    x_test_vector = best_model.named_steps["preprocess"].transform(x_test)
    x_train_vector = np.asarray(x_train_vector, dtype=float)
    x_test_vector = np.asarray(x_test_vector, dtype=float)

    save_txt("train.txt", x_train_vector, y_train)
    save_txt("test.txt", x_test_vector, y_test)

    model_features = transformed_feature_names(best_model, numeric_features, categorical_features)
    write_selected_features(
        "selected_features.txt",
        numeric_features,
        categorical_features,
        model_features,
    )

    metrics = {
        "model": "LinearSVC",
        "accuracy": float(test_accuracy),
        "AP": float(average_precision_score(y_test, scores)),
        "AUC": float(roc_auc_score(y_test, scores)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "random_state": RANDOM_STATE,
        "target_column": target_column,
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "vector_dim": int(x_train_vector.shape[1]),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nRaw numeric features:")
    for i, feature in enumerate(numeric_features, 1):
        print(f"{i}. {feature}")
    print("\nRaw categorical features:")
    for i, feature in enumerate(categorical_features, 1):
        print(f"{i}. {feature}")
    print(f"\nExpanded vector dimension: {x_train_vector.shape[1]}")
    print(f"Best C selected by train CV accuracy: {best_search['C']}")
    print(f"Best decision threshold selected by train CV accuracy: {best_search['threshold']:.6f}")
    print(f"CV accuracy: {best_search['cv_accuracy']:.6f}")
    print(f"Test accuracy: {test_accuracy:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
