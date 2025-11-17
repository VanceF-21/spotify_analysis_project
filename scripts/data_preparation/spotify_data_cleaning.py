#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spotify Data Processing Pipeline.

This script performs a complete data processing workflow.
It loads raw data, preprocesses it, performs deduplication,
maps cluster data, and engineers time-based and artist-based
features, saving the result as a single CSV file.
"""

import os
import re
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Set, Union, Dict, Any


SCRIPT_DIR = Path(__file__).resolve().parent


PROJECT_ROOT = SCRIPT_DIR.parent.parent


DATA_DIR = PROJECT_ROOT / "data" / "data_preparation"


# Input file (raw data)
INPUT_RAW_DATA: str = str(DATA_DIR / "Spotify_Dataset_V3.csv")
# Input file (k-means cluster mapping)
INPUT_KMEANS_MAP: str = str(DATA_DIR / "kmeans_clustered_data.csv")

# Final output file
FINAL_OUTPUT_FILE: str = str(DATA_DIR / "spotify_cleaned_data.csv")

# Output encoding
OUTPUT_ENCODING: str = "utf-8-sig"


#Pipeline Functions

def load_and_preprocess_data(input_path_str: str) -> pd.DataFrame:
    """
    Loads the raw data from a CSV file.

    Performs robust reading, data cleaning, type conversions,and initial feature engineering.
    Args: input_path_str: The file path to the raw input CSV.

    Returns: A preprocessed pandas DataFrame.
    """

    def smart_read_csv(path: str) -> pd.DataFrame:
        """Attempts to read a CSV using multiple encodings and separators."""
        encodings: List[str] = ["utf-8-sig", "utf-8", "latin-1"]
        # Try semicolon separator first
        for enc in encodings:
            try:
                df = pd.read_csv(path, sep=';', encoding=enc, low_memory=False)
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
        # Fallback to comma separator
        for enc in encodings:
            try:
                df = pd.read_csv(path, sep=',', encoding=enc, low_memory=False)
                return df
            except Exception:
                pass
        raise RuntimeError(f"Failed to read CSV with ';' or ',' using common encodings: {path}")

    df = smart_read_csv(input_path_str)

    #Basic Cleaning
    df.columns = [c.strip() for c in df.columns]

    if "Points (Ind for each Artist/Nat]" in df.columns and "Points (Ind for each Artist/Nat)" not in df.columns:
        df = df.rename(columns={"Points (Ind for each Artist/Nat]": "Points (Ind for each Artist/Nat)"})

    str_cols: List[str] = ["Title", "Artists", "Artist (Ind.)", "Nationality", "Continent", "Song URL", "id"]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "Date" not in df.columns:
        raise KeyError("`Date` column missing. Please check the source file.")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    critical_cols: List[str] = [c for c in ["Date", "Rank", "id"] if c in df.columns]
    df = df.dropna(subset=critical_cols)

    #Gentle Parsing for Count Columns
    def parse_count_column(s: pd.Series) -> pd.Series:
        """Robustly parses a string column to an integer count."""
        s = s.astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan}, regex=False)
        s = s.str.replace(",", ".", regex=False)  # Handle decimal comma
        s = s.str.replace(r"[^0-9.]", "", regex=True)  # Keep only digits/dots
        s = s.str.extract(r"(\d+(?:\.\d+)?)", expand=False)  # Extract first number
        num = pd.to_numeric(s, errors="coerce").round().astype("Int64")
        num = num.mask(num <= 0, np.nan)  # Non-positive counts are invalid
        return num

    for c in ["# of Artist", "# of Nationality"]:
        if c in df.columns:
            df[c] = parse_count_column(df[c])

    #Numeric Conversion and Clipping
    other_numeric: List[str] = [
        "Danceability", "Energy", "Loudness", "Speechiness",
        "Acousticness", "Instrumentalness", "Valence",
        "Points (Total)", "Points (Ind for each Artist/Nat)", "Rank"
    ]
    for c in other_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.drop_duplicates()

    if "Rank" in df.columns:
        df = df[(df["Rank"] >= 1) & (df["Rank"] <= 200)]

    # Fix Loudness values that are 1000x too large
    if "Loudness" in df.columns:
        mask_k = df["Loudness"].abs() >= 1000
        df.loc[mask_k, "Loudness"] = df.loc[mask_k, "Loudness"] / 1000.0

    # Clip [0, 1] features
    for c in ["Danceability", "Energy", "Speechiness", "Acousticness", "Instrumentalness", "Valence"]:
        if c in df.columns:
            df[c] = df[c].clip(0, 1)

    #Canonical Artist Key
    if "Artist (Ind.)" in df.columns:
        base_artist = df["Artist (Ind.)"].astype(str)
    elif "Artists" in df.columns:
        base_artist = df["Artists"].astype(str).str.split(",").str[0]
    else:
        base_artist = pd.Series("", index=df.index)

    df["Artist_Canon"] = (
        base_artist.str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    #Time Features
    dt = df["Date"]
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month

    # Create time keys in the specific order
    df["week"] = dt.dt.isocalendar().week.astype('Int64')
    df["dow"] = dt.dt.dayofweek  # Monday=0, Sunday=6
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)  # 1 for Sat/Sun, 0 otherwise

    df["yyyymm"] = dt.dt.to_period("M").astype(str)

    #Popularity Signals
    if 'Rank' in df.columns and 'rank_points' not in df.columns:
        df['rank_points'] = 201 - df["Rank"]

    if "Points (Total)" in df.columns:
        df["pop_points_total"] = df["Points (Total)"]
    elif 'Rank' in df.columns:
        df["pop_points_total"] = 201 - df["Rank"]

    if "Points (Ind for each Artist/Nat)" in df.columns:
        df["pop_points_artist"] = df["Points (Ind for each Artist/Nat)"]
    else:
        if "# of Artist" in df.columns:
            denom = df["# of Artist"].replace({0: np.nan})
            df["pop_points_artist"] = df["pop_points_total"] / denom
        else:
            df["pop_points_artist"] = df["pop_points_total"]

    #Column Dropping
    # Per analysis, no columns are dropped to match the target file.
    to_drop_user: List[str] = []
    to_drop_redundant: List[str] = []

    df = df.drop(columns=[c for c in to_drop_user + to_drop_redundant if c in df.columns], errors="ignore")

    return df


def deduplicate_by_day_and_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops intra-day duplicates based on 'Title'.

    Uses a (Date, Title) composite key to identify duplicates, keeping the first occurrence and preserving the original table order.

    Args:df: The preprocessed DataFrame.

    Returns:A deduplicated DataFrame.
    """

    DATE_COL: str = "Date"
    TITLE_COL: str = "Title"
    TITLE_NORMALIZE: str = "none"  # 'none' = exact match

    def make_title_key(s: pd.Series, mode: str) -> pd.Series:
        """Creates a normalized key for the title."""
        s = s.astype(str)
        if mode == "strip":
            return s.str.strip()
        elif mode == "strip_lower":
            return s.str.strip().str.lower()
        elif mode == "none":
            return s
        else:
            raise ValueError("TITLE_NORMALIZE must be one of: none | strip | strip_lower")

    def make_day_key(s: pd.Series) -> pd.Series:
        """Creates a normalized YYYY-MM-DD key from the Date column."""
        parsed = pd.to_datetime(s, errors="coerce")
        key = parsed.dt.strftime("%Y-%m-%d")
        return key.where(parsed.notna(), s.astype(str))

    #Main Logic
    if DATE_COL not in df.columns:
        raise KeyError(f"Deduplication Fail: Missing required column '{DATE_COL}'.")
    if TITLE_COL not in df.columns:
        raise KeyError(f"Deduplication Fail: Missing required column '{TITLE_COL}'.")

    day_key: pd.Series = make_day_key(df[DATE_COL])
    title_key: pd.Series = make_title_key(df[TITLE_COL], TITLE_NORMALIZE)

    # Create a boolean mask for duplicate rows (True = duplicate)
    dup_mask: pd.Series = pd.MultiIndex.from_arrays([day_key, title_key]).duplicated(keep='first')

    # Select only the rows that are *not* duplicates
    dedup_df: pd.DataFrame = df.loc[~dup_mask, df.columns].copy()

    return dedup_df


def map_cluster_data(main_df: pd.DataFrame, map_path_str: str) -> pd.DataFrame:
    """
    Maps k-means cluster data onto the main dataframe.

    Uses a robust, normalized (Title, Artists) key for matching and appends the cluster information (last 3 columns) from the map file.

    Args:main_df: The deduplicated main DataFrame. map_path_str: File path to the k-means cluster CSV.

    Returns: A DataFrame with cluster information merged.
    """

    map_path: Path = Path(map_path_str)

    TITLE_COL: str = "Title"
    ARTISTS_COL: str = "Artists"
    TITLE_ALIASES: Dict[str, str] = {}

    #Helper: Read Map File
    def read_map(path: Path) -> pd.DataFrame:
        """Robustly reads the mapping CSV file."""
        last_err: Union[Exception, None] = None
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            for args in [(dict(sep=",", encoding=enc, low_memory=False)),
                         (dict(sep=",", encoding=enc, engine="python")),
                         (dict(sep=None, engine="python", encoding=enc))]:
                try:
                    return pd.read_csv(path, **args)
                except Exception as e:
                    last_err = e
        raise RuntimeError(f"Failed to read map CSV: {path}\nLast error: {last_err}")

    #Helpers: Robust Key Normalization
    _dash_re = re.compile(r"[‐-‒–—―]")
    _space_re = re.compile(r"\s+")
    _strip_quotes_re = re.compile(r"^['\"“”‘’´`]+|['\"“”‘’´`]+$")
    _leading_year_apostrophe = re.compile(r"^['\"“”‘’´`]\s*(\d{2})(\b|[^0-9])")

    def _deaccent(text: str) -> str:
        """Removes diacritics/accents from a string."""
        return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

    def _fix_quotes(text: str) -> str:
        """Normalizes various quote characters to standard ' or "."""
        return (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"').replace("´", "'").replace("`", "'"))

    def _norm_basic(text: str) -> str:
        """Applies basic normalization: de-accent, fix quotes, fix dashes, lowercase, strip."""
        text = _fix_quotes(_deaccent(text))
        text = _dash_re.sub("-", text)
        text = _strip_quotes_re.sub("", text)
        text = _space_re.sub(" ", text).strip().lower()
        return text

    def normalize_title_for_key(s: pd.Series) -> pd.Series:
        """Normalizes a Title series for robust matching."""

        def _one(x: str) -> str:
            x = _norm_basic(x)
            x = _leading_year_apostrophe.sub(lambda m: m.group(1) + (m.group(2) if m.group(2) else ""), x)
            return TITLE_ALIASES.get(x, x)

        return s.astype(str).map(_one)

    _artist_split_re = re.compile(r"\s*(?:,|&| x | with | feat\.?| featuring | ft\.?)\s*", flags=re.IGNORECASE)

    def normalize_artists_for_key(s: pd.Series) -> pd.Series:
        """
        Normalizes an Artists series for robust matching. Splits, normalizes, de-duplicates, sorts, and re-joins.
        """

        def _one(x: str) -> str:
            x = _norm_basic(x)
            parts: List[str] = [p for p in _artist_split_re.split(x) if p.strip()]
            parts_set: Set[str] = sorted(set(_space_re.sub(" ", p).strip() for p in parts))
            return " | ".join(parts_set)

        return s.astype(str).map(_one)

    #Main Logic
    if not map_path.exists():
        raise FileNotFoundError(f"Cluster Mapping Fail: Mapping file not found: {map_path}")

    for c in (TITLE_COL, ARTISTS_COL):
        if c not in main_df.columns:
            raise KeyError(f"Cluster Mapping Fail: Main DataFrame missing '{c}'.")

    map_df = read_map(map_path)
    for c in (TITLE_COL, ARTISTS_COL):
        if c not in map_df.columns:
            raise KeyError(f"Cluster Mapping Fail: Mapping file '{map_path.name}' missing '{c}'.")

    if map_df.shape[1] < 3:
        raise ValueError("Cluster Mapping Fail: Mapping file has < 3 columns.")
    last3: List[str] = list(map_df.columns[-3:])

    # Create robust keys for both dataframes
    main_key = pd.DataFrame({
        "_k_title": normalize_title_for_key(main_df[TITLE_COL]),
        "_k_artists": normalize_artists_for_key(main_df[ARTISTS_COL]),
    })
    map_key = pd.DataFrame({
        "_k_title": normalize_title_for_key(map_df[TITLE_COL]),
        "_k_artists": normalize_artists_for_key(map_df[ARTISTS_COL]),
    })

    # Prepare map_df for merge, renaming new columns if they conflict
    map_keyed = map_key.copy()
    added_cols: List[str] = []
    for c in last3:
        newc = c if c not in main_df.columns else c + "__map"
        map_keyed[newc] = map_df[c]
        added_cols.append(newc)

    map_keyed = map_keyed.drop_duplicates(subset=["_k_title", "_k_artists"], keep="first")

    # Merge while preserving the main dataframe's order
    main_df["_row_order"] = np.arange(len(main_df))
    merged = pd.concat([main_df, main_key], axis=1).merge(
        map_keyed, on=["_k_title", "_k_artists"], how="left", sort=False, copy=False
    ).sort_values("_row_order", kind="mergesort")

    # Final columns = original columns + new mapped columns
    out_cols: List[str] = [c for c in main_df.columns if c != "_row_order"] + added_cols
    out_df: pd.DataFrame = merged[out_cols].copy()

    return out_df


def calculate_and_map_period_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and maps periodic cluster counts.

    Aggregates weekly, monthly, and quarterly occurrence counts for each cluster and maps these counts back to every row.
    Args: df: The DataFrame with cluster information mapped.

    Returns: A DataFrame with periodic counts included.
    """

    DATE_COL: str = "Date"
    CLUSTER_ID: str = "cluster_id"
    CLUSTER_LABEL: str = "cluster_label"


    # Check for remapped column names
    for c in (DATE_COL, CLUSTER_ID, CLUSTER_LABEL):
        if c not in df.columns:
            if c == CLUSTER_ID and 'cluster_id__map' in df.columns:
                CLUSTER_ID = 'cluster_id__map'
            elif c == CLUSTER_LABEL and 'cluster_label__map' in df.columns:
                CLUSTER_LABEL = 'cluster_label__map'
            elif c not in df.columns:
                raise KeyError(f"Period Count Fail: Missing required column '{c}'.")

    df["_row_order"] = np.arange(len(df))

    df["_date"] = pd.to_datetime(df[DATE_COL], errors="coerce")
    valid: pd.DataFrame = df.dropna(subset=["_date"]).copy()

    # Create time period keys
    valid["_week_start"] = valid["_date"].dt.to_period("W-MON").dt.start_time
    valid["_month"] = valid["_date"].dt.to_period("M").dt.start_time
    valid["_quarter"] = valid["_date"].dt.to_period("Q").dt.start_time

    by: List[str] = [CLUSTER_ID, CLUSTER_LABEL]

    # Calculate counts per cluster per period
    weekly_counts = (
        valid.groupby(by + ["_week_start"])
        .size().reset_index(name="cluster_week_count")
    )
    monthly_counts = (
        valid.groupby(by + ["_month"])
        .size().reset_index(name="cluster_month_count")
    )
    quarterly_counts = (
        valid.groupby(by + ["_quarter"])
        .size().reset_index(name="cluster_quarter_count")
    )

    # Add time keys to the main df for merging
    df["_week_start"] = df["_date"].dt.to_period("W-MON").dt.start_time
    df["_month"] = df["_date"].dt.to_period("M").dt.start_time
    df["_quarter"] = df["_date"].dt.to_period("Q").dt.start_time

    # Merge counts back to the main df
    merged = df.merge(
        weekly_counts, on=[CLUSTER_ID, CLUSTER_LABEL, "_week_start"], how="left", sort=False
    ).merge(
        monthly_counts, on=[CLUSTER_ID, CLUSTER_LABEL, "_month"], how="left", sort=False
    ).merge(
        quarterly_counts, on=[CLUSTER_ID, CLUSTER_LABEL, "_quarter"], how="left", sort=False
    )

    # Fill NaN (unmatched) counts with 0
    new_count_cols: List[str] = ["cluster_week_count", "cluster_month_count", "cluster_quarter_count"]
    for c in new_count_cols:
        merged[c] = merged[c].fillna(0).astype(int)

    # Clean up helper columns and restore order
    base_cols: List[str] = [c for c in df.columns if
                            c not in ["_row_order", "_date", "_week_start", "_month", "_quarter"]]
    out_cols: List[str] = base_cols + new_count_cols

    out_df: pd.DataFrame = merged.sort_values("_row_order", kind="mergesort")[out_cols].copy()

    return out_df


def calculate_artist_stats_and_save(df: pd.DataFrame, out_final_csv_path: str) -> None:
    """
    Calculates artist combo stats and saves the final file.

    Calculates mean/variance/count for audio features, grouped by the unique 'Artists' string. Maps these stats back and saves the final dataframe to a CSV file.

    Args: df: The DataFrame with periodic counts. out_final_csv_path: The file path to save the final CSV.
    """

    KEY_COL: str = "Artists"
    FEATURES: List[str] = ["Danceability", "Energy", "Loudness", "Speechiness", "Acousticness", "Instrumentalness",
                           "Valence"]
    USE_CANICAL_GROUP_KEY: bool = True
    DDOF: int = 0  # Population variance
    FILL_VAR_NA_WITH_ZERO: bool = False

    #Helpers: Artist Combo Key
    ARTIST_SPLIT_RE = re.compile(r"\s*(?:,|&| x | with | feat\.?| featuring | ft\.?)\s*", flags=re.IGNORECASE)

    def to_float(s: pd.Series) -> pd.Series:
        """Converts a string series to float, removing commas/%."""
        return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
                             errors="coerce")

    def _deaccent(text: str) -> str:
        """Removes diacritics/accents."""
        return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

    def _norm_name(name: str) -> str:
        """Normalizes a single artist name."""
        name = _deaccent(str(name))
        name = re.sub(r"\s+", " ", name).strip().lower()
        return name

    def make_combo_key(raw: str) -> str:
        """Creates a sorted, unique key for an artist string (e.g., "A, B & C")."""
        parts = [p.strip() for p in ARTIST_SPLIT_RE.split(str(raw)) if p.strip()]
        parts_set = sorted({_norm_name(p) for p in parts if p})
        if not parts_set:
            return ""
        return " | ".join(parts_set)


    if KEY_COL not in df.columns:
        raise KeyError(f"Artist Stats Fail: Missing key column '{KEY_COL}'.")

    # Find which features are actually available in the dataframe
    features_to_run: List[str] = []
    for f in FEATURES:
        if f in df.columns:
            features_to_run.append(f)

    if not features_to_run:
        # No features to process, just save the file as-is
        df.to_csv(out_final_csv_path, index=False, encoding=OUTPUT_ENCODING, sep=',')
        return

    df["_row_order__"] = np.arange(len(df))

    # Create a working copy for calculations
    work: pd.DataFrame = df[[KEY_COL] + features_to_run].copy()
    for f in features_to_run:
        work[f] = to_float(work[f])

    # Generate the group key
    if USE_CANICAL_GROUP_KEY:
        df["_combo_key__"] = df[KEY_COL].map(make_combo_key)
    else:
        df["_combo_key__"] = df[KEY_COL].astype(str)

    # Group and calculate stats
    grp = work.groupby(df["_combo_key__"])

    means = grp[features_to_run].mean().add_prefix("combo_mean_")
    vars_ = grp[features_to_run].var(ddof=DDOF).add_prefix("combo_var_")
    counts = grp[features_to_run].count().add_prefix("combo_n_")

    combo_stats = pd.concat([means, vars_, counts], axis=1).reset_index()
    combo_stats = combo_stats.rename(columns={combo_stats.columns[0]: "_combo_key__"})

    # Merge stats back to the main dataframe
    merged = df.merge(combo_stats, on="_combo_key__", how="left", sort=False)

    if FILL_VAR_NA_WITH_ZERO:
        for f in features_to_run:
            if "combo_var_" + f in merged.columns:
                merged["combo_var_" + f] = merged["combo_var_" + f].fillna(0)

    # Clean up helper columns and restore order
    new_cols: List[str] = list(col for col in merged.columns if
                               col.startswith("combo_mean_") or col.startswith("combo_var_") or col.startswith(
                                   "combo_n_"))
    base_cols: List[str] = [c for c in df.columns if c not in ["_row_order__", "_combo_key__"]]
    out_cols: List[str] = base_cols + new_cols

    out_df: pd.DataFrame = merged.sort_values("_row_order__", kind="mergesort")[out_cols].copy()

    # Save the final file
    out_df.to_csv(out_final_csv_path, index=False, encoding=OUTPUT_ENCODING, sep=',')


#Main Execution

def main() -> None:
    """
    Runs the complete data processing pipeline.
    """

    # Check for input files
    if not Path(INPUT_RAW_DATA).exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_RAW_DATA}")
    if not Path(INPUT_KMEANS_MAP).exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_KMEANS_MAP}")

    # Run the pipeline
    processed_df = load_and_preprocess_data(INPUT_RAW_DATA)

    deduped_df = deduplicate_by_day_and_title(processed_df)

    mapped_df = map_cluster_data(deduped_df, INPUT_KMEANS_MAP)

    counted_df = calculate_and_map_period_counts(mapped_df)

    calculate_artist_stats_and_save(counted_df, FINAL_OUTPUT_FILE)


if __name__ == "__main__":
    main()