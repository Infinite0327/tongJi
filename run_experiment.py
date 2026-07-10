import argparse
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


RANDOM_STATE = 42
TARGET_COLUMN = "宫内妊娠术后3-4周（胎心胎芽1）"
DEFAULT_EXCEL_PATHS = [
    Path("病历原始数据.xlsx"),
    Path("超声数据") / "病历原始数据.xlsx",
]

LEAKAGE_KEYWORDS = [
    "id",
    "编号",
    "序号",
    "病案",
    "病历",
    "住院号",
    "门诊号",
    "姓名",
    "名字",
    "name",
    "电话",
    "身份证",
    "标签",
    "label",
    "target",
    "妊娠",
    "结局",
    "结果",
    "术后",
    "hcg",
    "β",
    "beta",
    "胎心",
    "胎芽",
    "流产",
    "活产",
    "生化",
]

MISSING_MARKERS = {"", "-", "—", "–", "NA", "N/A", "nan", "NaN", "无"}
MIN_NUMERIC_COVERAGE = 0.95


def find_excel_path(cli_path):
    if cli_path:
        path = Path(cli_path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")
        return path

    for path in DEFAULT_EXCEL_PATHS:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find 病历原始数据.xlsx in the project root or 超声数据/."
    )


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

        sheet_path = Path("xl") / target
        sheet_path = str(sheet_path).replace("\\", "/")
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

    positive_tokens = ["1", "是", "有", "成功", "阳性", "妊娠", "怀孕", "临床妊娠", "见胎心"]
    negative_tokens = ["0", "否", "无", "失败", "阴性", "未妊娠", "未孕", "未见"]

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
    if is_target_column(column):
        return True
    return any(keyword.lower() in normalized for keyword in LEAKAGE_KEYWORDS)


def coerce_numeric_series(series):
    cleaned = series.replace(list(MISSING_MARKERS), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def numeric_candidate_columns(df):
    candidates = []
    excluded = []
    for column in df.columns:
        if is_leakage_or_identifier(column):
            excluded.append((column, "identifier/label/outcome leakage keyword"))
            continue

        converted = coerce_numeric_series(df[column])
        original_non_missing = df[column].replace(list(MISSING_MARKERS), np.nan).notna().sum()
        numeric_non_null = converted.notna().sum()
        coverage = numeric_non_null / len(df) if len(df) else 0.0
        if original_non_missing > 0 and numeric_non_null == original_non_missing and coverage >= MIN_NUMERIC_COVERAGE:
            candidates.append(column)
        else:
            excluded.append((column, "non-numeric, low numeric coverage, or partially non-numeric"))
    return candidates, excluded


def write_columns_preview(path, candidates, excluded):
    with open(path, "w", encoding="utf-8") as f:
        f.write("候选数值特征列（已排除 ID、姓名、标签、结局、HCG 等泄漏列）:\n")
        for i, column in enumerate(candidates, 1):
            f.write(f"{i}. {column}\n")
        f.write("\n被排除列:\n")
        for column, reason in excluded:
            f.write(f"- {column}: {reason}\n")


def minmax_transform(train_values, test_values):
    min_values = np.nanmin(train_values, axis=0)
    max_values = np.nanmax(train_values, axis=0)
    ranges = max_values - min_values
    ranges[ranges == 0] = 1.0
    return (train_values - min_values) / ranges, (test_values - min_values) / ranges


def median_impute_from_train(train_values, test_values):
    medians = np.nanmedian(train_values, axis=0)
    if np.isnan(medians).any():
        bad_columns = np.where(np.isnan(medians))[0].tolist()
        raise ValueError(f"Cannot impute columns with all-missing training values: {bad_columns}")

    train_filled = np.where(np.isnan(train_values), medians, train_values)
    test_filled = np.where(np.isnan(test_values), medians, test_values)
    return train_filled, test_filled


def save_txt(path, x_values, y_values):
    data = np.column_stack([x_values, y_values])
    np.savetxt(path, data, fmt="%.10g")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", default=None, help="Path to 病历原始数据.xlsx")
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

    candidates, excluded = numeric_candidate_columns(df)
    if len(candidates) != 14:
        write_columns_preview("columns_preview.txt", candidates, excluded)
        print(f"Found {len(candidates)} candidate numeric features, not exactly 14.")
        print("Wrote columns_preview.txt. Please confirm the 14 input features before training.")
        return 2

    try:
        from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for Linear SVM training and metric calculation. "
            "Install it with: python -m pip install scikit-learn"
        ) from exc

    selected_features = candidates
    with open("selected_features.txt", "w", encoding="utf-8") as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

    x_df = df[selected_features].apply(coerce_numeric_series)
    x = x_df.to_numpy(dtype=float)
    missing_count = int(np.isnan(x).sum())
    if missing_count:
        print(
            f"Found {missing_count} missing feature values; imputing with training-set medians after split."
        )

    if len(x_df) < 200:
        raise ValueError(f"Need at least 200 usable labeled rows for 160/40 split; got {len(x_df)}.")

    indices = np.arange(len(x_df))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=160,
        test_size=40,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2 else None,
    )

    x_train_raw = x[train_idx]
    x_test_raw = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    x_train_raw, x_test_raw = median_impute_from_train(x_train_raw, x_test_raw)
    x_train, x_test = minmax_transform(x_train_raw, x_test_raw)
    x_train = np.clip(x_train, 0.0, 1.0)
    x_test = np.clip(x_test, 0.0, 1.0)

    save_txt("train.txt", x_train, y_train)
    save_txt("test.txt", x_test, y_test)

    model = LinearSVC(random_state=RANDOM_STATE, max_iter=10000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    scores = model.decision_function(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "AP": float(average_precision_score(y_test, scores)),
        "AUC": float(roc_auc_score(y_test, scores)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "random_state": RANDOM_STATE,
        "model": "LinearSVC",
        "target_column": target_column,
        "selected_features": selected_features,
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n使用的14维特征名:")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")
    print(f"accuracy: {metrics['accuracy']:.6f}")
    print(f"AP: {metrics['AP']:.6f}")
    print(f"AUC: {metrics['AUC']:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
