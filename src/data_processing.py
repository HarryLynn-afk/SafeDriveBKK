"""
Data Processing Pipeline for Thai Road Accident Analysis

DATA SOURCE: Thai Ministry of Transport (2024)
- Real government accident records
- [NEW NUMBER] records with GPS coordinates
- Source: https://datagov.mot.go.th/en/dataset/roadaccident
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Bangkok bounding box
BKK_LAT_MIN, BKK_LAT_MAX = 13.5, 14.2
BKK_LON_MIN, BKK_LON_MAX = 100.3, 100.9

# Phuket bounding box
PKT_LAT_MIN, PKT_LAT_MAX = 7.5, 8.2
PKT_LON_MIN, PKT_LON_MAX = 98.0, 98.7

CITY_BOUNDS = {
    "bangkok": (BKK_LAT_MIN, BKK_LAT_MAX, BKK_LON_MIN, BKK_LON_MAX),
    "phuket": (PKT_LAT_MIN, PKT_LAT_MAX, PKT_LON_MIN, PKT_LON_MAX),
}


def load_raw_data(filepath):
    """Load CSV with proper encoding handling."""
    for encoding in ["utf-8-sig", "utf-8", "tis-620", "cp874"]:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"Loaded {len(df)} rows with encoding: {encoding}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Could not read {filepath} with any known encoding")


def clean_coordinates(df, city=None):
    """Remove rows with invalid or missing coordinates."""
    original = len(df)
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

    if city and city in CITY_BOUNDS:
        lat_min, lat_max, lon_min, lon_max = CITY_BOUNDS[city]
        df = df[
            (df["LATITUDE"] >= lat_min) & (df["LATITUDE"] <= lat_max) &
            (df["LONGITUDE"] >= lon_min) & (df["LONGITUDE"] <= lon_max)
        ]

    print(f"Coordinates cleaned: {original} -> {len(df)} rows ({original - len(df)} removed)")
    return df


def parse_datetime(df):
    """Parse Excel serial date/time values without dropping rows."""
    df = df.copy()
    excel_date = pd.to_numeric(df["วันที่เกิดเหตุ"], errors="coerce")
    excel_time = pd.to_numeric(df["เวลา"], errors="coerce")

    # Excel serial date: days since 1899-12-30
    df["date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(excel_date, unit="D")

    # Excel time: fraction of day (default to 12:00 when missing)
    raw_hours = excel_time * 24
    df["time_hours"] = np.floor(raw_hours).fillna(12).clip(0, 23).astype(int)
    df["time_minutes"] = np.floor((raw_hours * 60) % 60).fillna(0).clip(0, 59).astype(int)

    df["datetime"] = (
        df["date"]
        + pd.to_timedelta(df["time_hours"], unit="h")
        + pd.to_timedelta(df["time_minutes"], unit="m")
    )

    bad_date = df["date"].isna().sum()
    bad_time = excel_time.isna().sum()
    df.attrs["date_parse_errors"] = int(bad_date)
    df.attrs["time_parse_errors"] = int(bad_time)
    if bad_date > 0 or bad_time > 0:
        print(
            f"Warning: {bad_date} rows with unparseable date and "
            f"{bad_time} rows with unparseable time (rows kept)"
        )
    return df


def extract_features(df):
    """Create time-based features from datetime column."""
    df = df.copy()
    valid_datetime = df["datetime"].notna()

    df["hour"] = df["datetime"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["datetime"].dt.dayofweek.fillna(0).astype(int)  # 0=Mon, 6=Sun
    df["day_name"] = df["datetime"].dt.day_name().fillna("Unknown")
    df["month"] = df["datetime"].dt.month.fillna(0).astype(int)
    df["year"] = df["datetime"].dt.year.fillna(0).astype(int)
    df["week_number"] = df["datetime"].dt.isocalendar().week.fillna(0).astype(int)
    df["date"] = df["datetime"].dt.date

    df["is_weekend"] = valid_datetime & df["day_of_week"].isin([5, 6])
    df["is_rush_hour"] = valid_datetime & df["hour"].isin([7, 8, 9, 16, 17, 18, 19])

    valid_count = int(valid_datetime.sum())
    kept_count = len(df) - valid_count
    time_parse_errors = int(df.attrs.get("time_parse_errors", kept_count))
    print(f"Datetime after parsing: {valid_count}/{len(df)} rows valid")
    print(f"Rows kept despite time parsing failure: {time_parse_errors}")

    sample_cols = ["วันที่เกิดเหตุ", "เวลา", "datetime"]
    valid_sample = df.loc[valid_datetime, sample_cols].head(3)
    invalid_sample = df.loc[~valid_datetime, sample_cols].head(3)

    print("\nSample rows with valid datetime:")
    if not valid_sample.empty:
        print(valid_sample.to_string(index=False))
    else:
        print("None")

    print("\nSample rows without valid datetime:")
    if not invalid_sample.empty:
        print(invalid_sample.to_string(index=False))
    else:
        print("None")

    return df


def filter_city(df, city):
    """Filter data for a specific city."""
    city_map = {
        "bangkok": "กรุงเทพมหานคร",
        "phuket": "ภูเก็ต",
    }
    if city not in city_map:
        raise ValueError(f"Unknown city: {city}. Use: {list(city_map.keys())}")

    mask = df["จังหวัด"].str.contains(city_map[city], na=False)
    result = df[mask].copy()
    print(f"Filtered for {city}: {len(result)} rows")
    return result


def save_processed_data(df, output_path):
    """Save processed dataframe to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {output_path}")


def run_pipeline(input_path, output_dir="data/processed"):
    """Run the full cleaning pipeline."""
    print("=" * 60)
    print("Running data processing pipeline")
    print("=" * 60)

    df = load_raw_data(input_path)
    df = parse_datetime(df)
    df = extract_features(df)

    results = {}
    for city in ["bangkok", "phuket"]:
        print(f"\n--- Processing {city.title()} ---")
        city_df = filter_city(df, city)
        city_df = clean_coordinates(city_df, city)
        output_path = f"{output_dir}/{city}_accidents_clean.csv"
        save_processed_data(city_df, output_path)
        results[city] = city_df

    # Generate data quality report
    report = generate_quality_report(df, results)
    report_path = f"{output_dir}/data_quality_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nData quality report saved to {report_path}")

    return results


def generate_quality_report(full_df, city_dfs):
    """Generate a text report on data quality."""
    lines = [
        "=" * 60,
        "DATA QUALITY REPORT",
        "=" * 60,
        f"\nTotal raw records: {len(full_df)}",
        f"Date range: {full_df['datetime'].min()} to {full_df['datetime'].max()}",
        "",
    ]

    for city, df in city_dfs.items():
        lines.extend([
            f"--- {city.title()} ---",
            f"  Records after cleaning: {len(df)}",
            f"  GPS bounds: lat [{df['LATITUDE'].min():.4f}, {df['LATITUDE'].max():.4f}]",
            f"              lon [{df['LONGITUDE'].min():.4f}, {df['LONGITUDE'].max():.4f}]",
            f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}",
            f"  Missing values:",
        ])
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                lines.append(f"    {col}: {missing} ({100*missing/len(df):.1f}%)")
        if all(df[col].isna().sum() == 0 for col in df.columns):
            lines.append("    None")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    run_pipeline("data/raw/accident2024_REAL.csv")
