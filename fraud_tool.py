import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_input_file(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported input type: {suffix}. Use .csv or .xlsx")

    return df


def risk_band_from_score(score: int) -> str:
    if score >= 8:
        return "High"
    if score >= 4:
        return "Medium"
    return "Low"


# -----------------------------
# Core processing
# -----------------------------
def clean_and_standardise(df: pd.DataFrame) -> pd.DataFrame:
    df = standardise_column_names(df)

    required = ["customer_id", "quote_id", "quote_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["customer_id"] = df["customer_id"].astype("string").str.strip()
    df["quote_id"] = df["quote_id"].astype("string").str.strip()
    df["quote_time"] = safe_to_datetime(df["quote_time"])

    if "session_id" in df.columns:
        df["session_id"] = df["session_id"].astype("string").str.strip()
    else:
        df["session_id"] = pd.NA

    for col in ["occupation", "postcode"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    for col in ["age", "mileage", "premium"]:
        if col in df.columns:
            df[col] = safe_to_numeric(df[col])

    df = df.dropna(subset=["customer_id", "quote_time"]).copy()
    return df


def ensure_session_ids(df: pd.DataFrame, session_gap_minutes: int = 30) -> pd.DataFrame:
    df = df.sort_values(["customer_id", "quote_time", "quote_id"]).copy()

    if df["session_id"].isna().all():
        gap = df.groupby("customer_id")["quote_time"].diff()
        new_session = (gap > pd.Timedelta(minutes=session_gap_minutes)).fillna(True)
        session_num = new_session.groupby(df["customer_id"]).cumsum()
        df["session_id"] = (
            df["customer_id"].astype(str)
            + "_S"
            + session_num.astype(int).astype(str)
        )

    return df


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["customer_id", "session_id", "quote_time", "quote_id"]).copy()

    for col in ["occupation", "mileage", "premium"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["prev_occupation"] = df.groupby(["customer_id", "session_id"])["occupation"].shift(1)
    df["prev_mileage"] = df.groupby(["customer_id", "session_id"])["mileage"].shift(1)
    df["prev_premium"] = df.groupby(["customer_id", "session_id"])["premium"].shift(1)

    df["occupation_changed"] = df["occupation"] != df["prev_occupation"]
    df["mileage_changed"] = df["mileage"] != df["prev_mileage"]

    df["premium_drop_after_occupation_change"] = (
        df["occupation_changed"]
        & df["premium"].notna()
        & df["prev_premium"].notna()
        & (df["premium"] < df["prev_premium"])
    )

    df["premium_drop_after_mileage_change"] = (
        df["mileage_changed"]
        & df["premium"].notna()
        & df["prev_premium"].notna()
        & (df["premium"] < df["prev_premium"])
    )

    session_counts = df.groupby(["customer_id", "session_id"]).size()
    session_span = df.groupby(["customer_id", "session_id"])["quote_time"].agg(
        lambda s: (s.max() - s.min()).total_seconds() / 60
    )

    df = df.join(session_counts.rename("quotes_in_session"), on=["customer_id", "session_id"])
    df = df.join(session_span.rename("session_span_minutes"), on=["customer_id", "session_id"])

    df["high_frequency_session"] = (
        (df["quotes_in_session"] >= 6) & (df["session_span_minutes"] <= 10)
    )

    df["risk_score"] = 0
    df.loc[df["occupation_changed"], "risk_score"] += 2
    df.loc[df["mileage_changed"], "risk_score"] += 2
    df.loc[df["premium_drop_after_occupation_change"], "risk_score"] += 3
    df.loc[df["premium_drop_after_mileage_change"], "risk_score"] += 3
    df.loc[df["high_frequency_session"], "risk_score"] += 2

    df["risk_band"] = df["risk_score"].apply(lambda x: risk_band_from_score(int(x)))
    return df


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Quote manipulation decision support tool"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_input_file(input_path)
    df = clean_and_standardise(df)
    df = ensure_session_ids(df)
    df = apply_rules(df)

    df.to_csv(output_dir / "quote_level_fraud_analysis.csv", index=False)

    print("Analysis complete.")
    print(f"Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
