import pandas as pd
import re
from pathlib import Path

# data
CSV_PATH = r"D:/AAA!!2025-2026Edin/aml/project/spotify_dedup_by_day_title__with_combo_stats.csv"
OUT_DIR = Path(CSV_PATH).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_ARTISTS = 8
TOP_K_FEATURES_PER_ARTIST = 3

# 7 Single Audio Features
TARGET_FEATURES = [
    "Danceability", "Energy", "Loudness", "Speechiness",
    "Acousticness", "Instrumentalness", "Valence"
]

# HELPER FUNCTIONS 
def pick_first(cols, candidates):
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None

def resolve_feature_columns(cols, targets):
    """
    Select the most reasonable actual column for each target feature:
    Priority: Exact match ignoring case;
    Otherwise: Partial match, preferring 'raw values/means', excluding var/std/count/share/rank/points.
    """
    resolved = {}
    for feat in targets:
        exact = [c for c in cols if c.lower() == feat.lower()]
        if exact:
            resolved[feat] = exact[0]
            continue
        # Otherwise: Partial match
        cands = [c for c in cols if feat.lower() in c.lower()]
        def score(name):
            n = name.lower()
            bad = any(k in n for k in ["var", "std", "count", "share", "rank", "points", "_id", "uri"])
            prefer_mean = "mean" in n
            # Prefer 'raw-like columns' , then 'mean'
            raw_like = not ("mean" in n or "combo" in n)
            return (
                0 if bad else 1,
                2 if raw_like else (1 if prefer_mean else 0),
                -len(n)
            )
        if cands:
            cands.sort(key=score, reverse=True)
            resolved[feat] = cands[0]
        else:
            resolved[feat] = None
    return resolved

def clean_artist_tokens(s):
    if pd.isna(s): return []
    parts = re.split(r'\s*(,|&|\bx\b|feat\.?|featuring|with|and|\+|;|\|)\s*', str(s), flags=re.I)
    return [p.strip() for p in parts if p and p.lower() not in {
        ',', '&', 'x', 'feat', 'feat.', 'featuring', 'with', 'and', '+', ';', '|'
    }]

def make_track_key(df, cols):
    """Track-level deduplication key: Priority ID/URI; otherwise Title + Artist."""
    id_col = pick_first(cols, ["id", "track_id", "spotify_id", "uri", "track_uri"])
    if id_col is not None:
        return df[id_col].astype(str)
    
    title_col = pick_first(cols, ["Track Name", "Track", "Title", "Song", "track_name"])
    artists_combo = pick_first(cols, ["Artists", "Artist(s)", "All_Artists"])
    artist_single = pick_first(cols, ["Artist", "Artist_Name", "PrimaryArtist", "Artist_Canon"])
    
    if title_col is not None and (artists_combo is not None or artist_single is not None):
        acol = artists_combo if artists_combo is not None else artist_single
        return (df[title_col].map(lambda x: str(x).lower().strip())
                + " || " + df[acol].map(lambda x: str(x).lower().strip()))
    
    return pd.Series(range(len(df)), dtype=str)

def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH, engine="python", sep=None)
    df.columns = [c.strip() for c in df.columns]
    cols = list(df.columns)

    # 1. Resolve Feature Columns
    feat_map = resolve_feature_columns(cols, TARGET_FEATURES)
    missing = [k for k, v in feat_map.items() if v is None]
    if missing:
        raise ValueError(f"Features not found in table: {missing}\nCurrent mapping: {feat_map}")

  # 2. Generate Track Key
    df["TrackKey"] = make_track_key(df, cols)

  # 3. Prepare Data: Rename features to standard names and ensure numeric
    # We create a working copy for global stats first
    global_cols = ["TrackKey"] + list({v for v in feat_map.values()})
    df_global = df[global_cols].rename(columns={feat_map[k]: k for k in feat_map}).copy()
    for f in TARGET_FEATURES:
        df_global[f] = pd.to_numeric(df_global[f], errors="coerce")

 # PART 1: GLOBAL STATISTICS ACROSS ALL ARTISTS (From Script 2)
    print("\n Processing Part 1: Global Feature Statistics")

  # Deduplicate at track level 
    track_level_global = (df_global.groupby("TrackKey", as_index=False)
                          .agg({f: "mean" for f in TARGET_FEATURES}))

   # Calculate Global Means
    overall_means = (track_level_global[TARGET_FEATURES]
                     .mean(numeric_only=True)
                     .to_frame("Mean")
                     .reset_index()
                     .rename(columns={"index": "Feature"}))
    # Count valid tracks per feature
    overall_counts = (track_level_global[TARGET_FEATURES]
                      .count()
                      .to_frame("N_Tracks")
                      .reset_index()
                      .rename(columns={"index": "Feature"}))

    overall = overall_means.merge(overall_counts, on="Feature", how="left") \
                           .sort_values("Mean", ascending=False) \
                           .reset_index(drop=True)

  # Top 3 and Bottom 3 Global Features
    top3_global = overall.nlargest(3, "Mean").reset_index(drop=True)
    bottom3_global = overall.nsmallest(3, "Mean").reset_index(drop=True)

   # Export Global Stats
    overall_path = OUT_DIR / "overall_feature_means_across_all_artists.csv"
    top3_path = OUT_DIR / "overall_feature_means_top3.csv"
    bottom3_path = OUT_DIR / "overall_feature_means_bottom3.csv"

    overall.to_csv(overall_path, index=False, encoding="utf-8")
    top3_global.to_csv(top3_path, index=False, encoding="utf-8")
    bottom3_global.to_csv(bottom3_path, index=False, encoding="utf-8")

 #   print(f"[Saved] Global Means: {overall_path}")
    print(f"[Saved] Global Top 3: {top3_path}")
    print(f"[Saved] Global Bottom 3: {bottom3_path}")

    # Print Global Stats
    print("\n=== Global Means of 7 Features (All Artists) ===")
    print(overall.to_string(index=False))
    print("\n=== Global Top 3 Features ===")
    print(top3_global.to_string(index=False))
    print("\n=== Global Bottom 3 Features ===")
    print(bottom3_global.to_string(index=False))


# PART 2: TOP ARTISTS SPECIFIC ANALYSIS (From Script 1)
    print("\n--- Processing Part 2: Top Artists Analysis ---")

    # Identify necessary columns for artist processing
    artist_col_single = pick_first(cols, ["Artist", "Artist_Name", "PrimaryArtist", "Artist_Canon"])
    artists_col_combo = pick_first(cols, ["Artists", "Artist(s)", "All_Artists"])
    pop_points_col = pick_first(cols, ["pop_points_artist", "pop points artist"])

    if (artist_col_single is None) and (artists_col_combo is None):
        raise ValueError("Artist column ('Artist' or 'Artists') not found.")
    if pop_points_col is None:
        raise ValueError("Column 'pop_points_artist' not found.")

    # Prepare Base Table for Artists
    # We need TrackKey, PopPoints, and the features
    base_cols = ["TrackKey", pop_points_col] + list({v for v in feat_map.values()})
    
    if artist_col_single is not None:
        base = df[[artist_col_single] + base_cols].rename(
            columns={artist_col_single: "Artist", pop_points_col: "PopPoints",
                     **{feat_map[k]: k for k in feat_map}}
        )
    else:
        # Explode artists if only combo column exists
        tmp = df[[artists_col_combo] + base_cols].rename(
            columns={artists_col_combo: "Artists", pop_points_col: "PopPoints",
                     **{feat_map[k]: k for k in feat_map}}
        )
        tmp["ArtistList"] = tmp["Artists"].apply(clean_artist_tokens)
        base = tmp.explode("ArtistList", ignore_index=True).rename(columns={"ArtistList": "Artist"})
        base = base.drop(columns=["Artists"])

    # Clean Artist Data
    base["Artist"] = base["Artist"].fillna("").astype(str).str.strip()
    base = base[base["Artist"] != ""].copy()
    base["PopPoints"] = pd.to_numeric(base["PopPoints"], errors="coerce").fillna(0)
    for f in TARGET_FEATURES:
        base[f] = pd.to_numeric(base[f], errors="coerce")

 # Step 2.1: Select Top K Artists by Total Pop Points
    top_artists = (base.groupby("Artist", as_index=False)["PopPoints"].sum()
                  .sort_values("PopPoints", ascending=False)
                  .head(TOP_K_ARTISTS).reset_index(drop=True)
                  .rename(columns={"PopPoints": "TotalPopPoints"}))

    top_artists_path = OUT_DIR / "top8_artists_by_pop_points.csv"
    top_artists.to_csv(top_artists_path, index=False, encoding="utf-8")
    print(f"[Saved] Top {TOP_K_ARTISTS} Artists (by pop points): {top_artists_path}")

    # Step 2.2: Filter data for these artists
    base_top = base[base["Artist"].isin(top_artists["Artist"])].copy()

 # Step 2.3: Track Deduplication -> Artist Means
    # Mean of features for duplicate (Artist, TrackKey)
    track_level_artist = (base_top.groupby(["Artist", "TrackKey"], as_index=False)
                          .agg({f: "mean" for f in TARGET_FEATURES}))

    # Aggregate by Artist
    artist_feature_means = (track_level_artist.groupby("Artist", as_index=False)
                            .agg({f: "mean" for f in TARGET_FEATURES}))

    # Convert to Long Format
    long = artist_feature_means.melt(
        id_vars=["Artist"], value_vars=TARGET_FEATURES,
        var_name="Feature", value_name="Mean"
    )

    # Merge Total Points for sorting
    long = long.merge(top_artists, on="Artist", how="left")

    # Rank features within each artist
    long["RankWithinArtist"] = (long.groupby("Artist")["Mean"]
                                .rank(method="first", ascending=False).astype(int))

  # Select Top 3 features per artist
    top3_features_long = (long[long["RankWithinArtist"] <= TOP_K_FEATURES_PER_ARTIST]
                          .sort_values(["TotalPopPoints", "Artist", "RankWithinArtist"],
                                       ascending=[False, True, True])
                          .reset_index(drop=True))

    # Create Wide Format 
    def to_wide(df_artist):
        df_sorted = df_artist.sort_values(["RankWithinArtist", "Feature"])
        row = {
            "Artist": df_artist["Artist"].iloc[0],
            "TotalPopPoints": int(df_artist["TotalPopPoints"].iloc[0])
        }
        for i, (_, r) in enumerate(df_sorted.head(TOP_K_FEATURES_PER_ARTIST).iterrows(), start=1):
            row[f"Feature{i}"] = r["Feature"]
            row[f"Feature{i}_Mean"] = round(float(r["Mean"]), 4) if pd.notna(r["Mean"]) else None
        return row

    wide_rows = [to_wide(sub) for _, sub in long.groupby("Artist")]
    wide = pd.DataFrame(wide_rows).sort_values("TotalPopPoints", ascending=False).reset_index(drop=True)

  # Export Artist Stats
    out_long_all7 = OUT_DIR / "top8_artists_all7_feature_means_long.csv"
    long.sort_values(["TotalPopPoints", "Artist", "Mean"], ascending=[False, True, False]) \
        .to_csv(out_long_all7, index=False, encoding="utf-8")
    
    out_top3_long = OUT_DIR / "top8_artists_top3_single_features_long.csv"
    top3_features_long.to_csv(out_top3_long, index=False, encoding="utf-8")
    
    out_wide = OUT_DIR / "top8_artists_top3_single_features_wide.csv"
    wide.to_csv(out_wide, index=False, encoding="utf-8")

    print(f"[Saved] Artist Long Table (All 7 features): {out_long_all7}")
    print(f"[Saved] Artist Long Table (Top 3 features): {out_top3_long}")
    print(f"[Saved] Artist Wide Table (Top 3 features): {out_wide}")

    # Print Artist Stats
    print(f"\n=== Top {TOP_K_ARTISTS} Artists (Sum of Pop Points) ===")
    print(top_artists.to_string(index=False))

    print(f"\n=== Top {TOP_K_FEATURES_PER_ARTIST} Features per Top Artist ===")
    for _, row in wide.iterrows():
        print(f"\n[{row['Artist']}] PopPoints={row['TotalPopPoints']}")
        for i in range(1, TOP_K_FEATURES_PER_ARTIST + 1):
            f = row.get(f"Feature{i}")
            m = row.get(f"Feature{i}_Mean")
            if pd.notna(f):
                print(f"  #{i}: {f:15s} mean={m:.4f}")

if __name__ == "__main__":
    main()