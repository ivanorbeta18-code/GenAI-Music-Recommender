"""
Data loading, cleaning, mood-scoring, and filtering helpers for the music
recommender.

This dataset has no release-year column, so instead of a "favorite time
period" filter, each track gets a **mood** label derived from its audio
features (valence = musical positivity, energy = intensity), and
**popularity** stands in as the other axis for narrowing candidates.

Also includes a lightweight, keyword-based relevance gate used by the
chatbot so it only calls the Hugging Face model for questions that are
actually about the dataset.
"""

import re

import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = ["track_name", "artist_name", "genre"]

OPTIONAL_NUMERIC_COLUMNS = [
    "popularity", "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "liveness", "speechiness",
    "loudness", "duration_ms",
]

# Quadrants of the valence/energy plane -> a human mood label.
MOOD_QUADRANTS = {
    ("high", "high"): "Happy & Energetic",
    ("high", "low"): "Chill & Content",
    ("low", "high"): "Angry & Intense",
    ("low", "low"): "Sad & Mellow",
}


@st.cache_data(show_spinner=False)
def load_dataset(file_or_path) -> pd.DataFrame:
    """Load a Spotify-style CSV dataset, clean it, and add a `mood` column."""
    df = pd.read_csv(file_or_path)

    # Drop a stray leading index column some exports include (e.g. an
    # unnamed "," first column).
    if df.columns[0].strip() == "" or str(df.columns[0]).lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])

    # Normalize column names (handles common Kaggle Spotify dataset variants).
    rename_map = {
        "artists": "artist_name",
        "artist": "artist_name",
        "track_genre": "genre",
        "genres": "genre",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. "
            f"Expected at least: {REQUIRED_COLUMNS}"
        )

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df["track_name"] = df["track_name"].astype(str).str.strip()
    df["artist_name"] = df["artist_name"].astype(str).str.strip()
    df["genre"] = df["genre"].astype(str).str.strip()

    for col in OPTIONAL_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Mood profile from valence + energy -------------------------------
    if "valence" in df.columns and "energy" in df.columns:
        df["valence"] = df["valence"].fillna(df["valence"].median())
        df["energy"] = df["energy"].fillna(df["energy"].median())
        v_level = df["valence"].apply(lambda v: "high" if v >= 0.5 else "low")
        e_level = df["energy"].apply(lambda e: "high" if e >= 0.5 else "low")
        df["mood"] = [MOOD_QUADRANTS[(v, e)] for v, e in zip(v_level, e_level)]
    else:
        df["mood"] = "Unknown"

    if "popularity" in df.columns:
        df["popularity"] = df["popularity"].fillna(0).clip(0, 100)
    else:
        df["popularity"] = 0

    df = df.drop_duplicates(subset=["track_name", "artist_name"])
    df = df.reset_index(drop=True)
    return df


def filter_candidates(
    df: pd.DataFrame,
    genres: list[str],
    artists: list[str],
    moods: list[str],
    popularity_range: tuple[int, int],
    max_candidates: int = 60,
) -> pd.DataFrame:
    """
    Score every track by how well it matches the user's stated favorites
    (genre, artist, mood, popularity band), then return the top candidates.
    This narrows the ~114k-row dataset down to a shortlist that's cheap to
    hand to the GenAI step.
    """
    scored = df.copy()
    scored["match_score"] = 0.0

    if genres:
        scored["match_score"] += scored["genre"].isin(genres).astype(float) * 2.0

    if artists:
        scored["match_score"] += scored["artist_name"].isin(artists).astype(float) * 3.0

    if moods:
        scored["match_score"] += scored["mood"].isin(moods).astype(float) * 2.0

    lo, hi = popularity_range
    in_range = scored["popularity"].between(lo, hi)
    scored.loc[in_range, "match_score"] += 1.0
    # Small bonus so ties lean toward better-known tracks.
    scored["match_score"] += (scored["popularity"] / 100.0) * 0.5

    candidates = scored[scored["match_score"] > 0]
    if candidates.empty:
        candidates = scored

    candidates = candidates.sort_values("match_score", ascending=False)
    return candidates.head(max_candidates).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chatbot relevance gate
# ---------------------------------------------------------------------------

MUSIC_KEYWORDS = {
    "song", "songs", "track", "tracks", "artist", "artists", "genre", "genres",
    "recommend", "recommendation", "recommendations", "suggest", "suggestion",
    "mood", "moods", "popular", "popularity", "unheard", "discover", "hidden",
    "gem", "gems", "similar", "playlist", "listen", "music", "album", "albums",
    "tempo", "dance", "danceable", "danceability", "energy", "energetic",
    "valence", "chill", "sad", "happy", "angry", "mellow", "intense",
    "dataset", "chart", "top", "best", "underrated", "trending", "acoustic",
    "acousticness", "instrumental", "instrumentalness", "loud", "loudness",
    "explicit", "random",
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "who", "what", "when",
    "where", "why", "how", "do", "does", "did", "of", "for", "and", "or",
    "to", "in", "on", "at", "me", "my", "you", "your", "i", "it", "this",
    "that", "please", "can", "could", "would", "should", "tell", "give",
    "about", "some", "any", "will", "not",
}


@st.cache_data(show_spinner=False)
def build_vocab(df: pd.DataFrame) -> dict:
    """Builds the two pieces the chatbot's relevance gate checks against:

    - `tokens`: single-word vocabulary (music keywords + genre words).
      Genres are a small (~100), music-specific set, so single-word
      overlap is safe here.
    - `artist_phrases`: full lowercase artist names, restricted to
      multi-word names. Individual words from artist names are deliberately
      *not* added to `tokens` — with 30k+ artists in a real dataset, common
      English words (e.g. "weather", "president", "united") show up as
      single-word artist names or fragments and would make almost any
      sentence look "dataset-related". Matching the full multi-word phrase
      instead keeps that specific enough to be a real signal.
    """
    tokens = set(MUSIC_KEYWORDS)
    for g in df["genre"].dropna().unique():
        tokens.update(re.findall(r"[a-z0-9']+", str(g).lower()))
    tokens -= STOPWORDS

    artist_phrases = {
        a.lower() for a in df["artist_name"].dropna().unique()
        if isinstance(a, str) and " " in a.strip() and len(a.strip()) >= 6
    }

    return {"tokens": tokens, "artist_phrases": artist_phrases}


def is_dataset_related(query: str, vocab: dict) -> bool:
    """Heuristic relevance check: does the question touch music/dataset
    vocabulary (a genre, a music-domain keyword like 'recommend'/'mood'/
    'popular', or a full artist name)? If not, the chatbot should decline
    instead of calling the model."""
    q_lower = query.lower()
    words = set(re.findall(r"[a-z0-9']+", q_lower)) - STOPWORDS
    if not words:
        return False

    if words & vocab["tokens"]:
        return True

    return any(phrase in q_lower for phrase in vocab["artist_phrases"])