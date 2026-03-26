import pandas as pd
import re

# ---------- Normalization ----------
def normalize(text):
    if pd.isna(text):
        return ""
    text = text.lower()

    # remove ( ... ) and [ ... ]
    text = re.sub(r"\(.*?\)|\[.*?\]", "", text)

    # remove feat / ft
    text = re.sub(r"\b(feat|ft|featuring)\b.*", "", text)

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def make_key(df, artist_col, track_col):
    return (
        df[artist_col].apply(normalize) + "|" +
        df[track_col].apply(normalize)
    )


# ---------- Load ----------
charts = pd.read_csv("../datasets/charts.csv")
spotify_tracks = pd.read_csv("../datasets/spotify_tracks.csv")
spotify_clean = pd.read_csv("../datasets/spotify_data_clean.csv")
track_data = pd.read_csv("../datasets/track_data_final.csv")
sample = pd.read_csv("../datasets/sample_100k.csv")


# ---------- Build keys ----------
charts["key"] = make_key(charts, "artist", "song")
spotify_tracks["key"] = make_key(spotify_tracks, "artist_name", "track_name")
spotify_clean["key"] = make_key(spotify_clean, "artist_name", "track_name")
track_data["key"] = make_key(track_data, "artist_name", "track_name")
sample["key"] = make_key(sample, "artist_name", "track_name")


# ---------- Merge (progressive) ----------
df = charts.merge(spotify_tracks, on="key", how="left", suffixes=("", "_sp"))
df = df.merge(spotify_clean, on="key", how="left", suffixes=("", "_clean"))
df = df.merge(track_data, on="key", how="left", suffixes=("", "_track"))
df = df.merge(sample, on="key", how="left", suffixes=("", "_sample"))


# ---------- Optional: duration filter (improves correctness) ----------
def duration_match(a, b, tol=3000):
    if pd.isna(a) or pd.isna(b):
        return True
    return abs(a - b) <= tol

if "duration_ms" in df.columns and "track_duration_ms" in df.columns:
    df = df[df.apply(lambda r: duration_match(
        r.get("duration_ms"),
        r.get("track_duration_ms")
    ), axis=1)]


# ---------- Save ----------
df.to_csv("../datasets/merged_tracks.csv", index=False)