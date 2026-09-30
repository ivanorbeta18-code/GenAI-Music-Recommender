"""
GenAI layer — a single Hugging Face instruction-tuned model powers both
features of the app:

- **Recommendation reasoning** — given a pandas-narrowed candidate
  shortlist, the model picks + explains the final track recommendations.
- **Dataset chatbot** — answers free-form user questions about the
  dataset (e.g. "recommend me a chill artist", "give me a random unheard
  pop track"). It now receives the recent conversation history, so
  follow-ups like "give me more of those" make sense. The app only calls
  it for questions that pass `data_utils.is_dataset_related()` (or look
  like a follow-up to an earlier answer) — everything else gets a canned
  refusal instead of an API call.

The Hugging Face access token is never typed into the UI. It's read once
from `.streamlit/secrets.toml` (see `get_hf_token()` below), which is
git-ignored so it never leaves this machine / deployment target.
"""

import json
import re

import streamlit as st

# Any instruction-tuned chat model on the Hugging Face Inference API works
# here; swap this if your token doesn't have access to it. Qwen2.5-Instruct
# is ungated (no license click-through needed) and follows "answer only
# from this context" instructions well. For a faster/lighter option under
# free-tier rate limits, try "HuggingFaceTB/SmolLM2-1.7B-Instruct" instead.
DEFAULT_HF_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Max tokens the model may generate per chatbot answer. Raised from 300 so
# long lists don't get cut off mid-sentence.
CHAT_MAX_TOKENS = 1500

# How many past messages (user + assistant combined) are sent back to the
# model so it can resolve "those", "that artist", "more", etc.
HISTORY_TURNS = 6

REFUSAL_MESSAGE = (
    "I can only help with questions about this specific dataset — things "
    "like artist or genre recommendations, mood-based picks, or surfacing "
    "unheard tracks *from the tracks loaded in this app*. I don't have "
    "knowledge of music outside this dataset. Try asking something like "
    "\"recommend me a chill indie artist\" or \"give me a random unheard "
    "pop track\"."
)


def get_hf_token() -> str:
    """Reads the Hugging Face token embedded in the app's own secrets
    config — never from a user-facing text field. See
    `.streamlit/secrets.toml.example` for the expected key name."""
    try:
        return st.secrets["HF_TOKEN"]
    except (KeyError, FileNotFoundError):
        return ""


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Hugging Face: candidate shortlist -> final recommendations + reasons
# ---------------------------------------------------------------------------

def build_recommend_prompt(genres, artists, moods, popularity_range, candidates_df, n_recommend: int) -> str:
    lo, hi = popularity_range
    candidate_records = candidates_df[
        [c for c in ["track_name", "artist_name", "genre", "mood", "popularity"]
         if c in candidates_df.columns]
    ].to_dict(orient="records")

    prompt = f"""You are a music recommendation assistant working from a fixed Spotify-style
dataset. You must ONLY recommend tracks that appear in the candidate list below —
never invent a track, artist, or album that isn't in the list.

User's stated favorites:
- Favorite genre(s): {', '.join(genres) if genres else 'no strong preference'}
- Favorite artist(s): {', '.join(artists) if artists else 'no strong preference'}
- Favorite mood(s): {', '.join(moods) if moods else 'no strong preference'}
- Popularity range they're open to (0=obscure, 100=mainstream): {lo}-{hi}

Candidate tracks from the dataset (JSON):
{json.dumps(candidate_records, ensure_ascii=False)}

Pick the {n_recommend} best tracks for this user from the candidate list above.
Favor variety (don't just return the same artist repeatedly) while still respecting
their stated favorites. For each pick, give a one-sentence reason that references
their actual stated preferences (genre, artist, or mood).

Respond ONLY with valid JSON, no preamble, no markdown fences, in this exact shape:
{{
  "recommendations": [
    {{"track_name": "...", "artist_name": "...", "genre": "...", "mood": "...", "reason": "..."}}
  ],
  "overall_note": "one short sentence summarizing the taste profile you detected"
}}
"""
    return prompt


def get_recommendations(hf_model, genres, artists, moods, popularity_range, candidates_df, n_recommend=8):
    """Calls the Hugging Face model with the shortlisted candidates and
    returns parsed JSON. Uses the token embedded via `get_hf_token()`."""
    from huggingface_hub import InferenceClient

    if candidates_df.empty:
        return {"recommendations": [], "overall_note": "No candidate tracks matched the filters."}

    hf_token = get_hf_token()
    if not hf_token:
        raise RuntimeError(
            "No Hugging Face token found. Add HF_TOKEN to "
            ".streamlit/secrets.toml (see secrets.toml.example)."
        )

    prompt = build_recommend_prompt(genres, artists, moods, popularity_range, candidates_df, n_recommend)
    client = InferenceClient(model=hf_model, token=hf_token)
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.6,
    )
    text = _strip_json_fence(response.choices[0].message.content)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        st.error("The AI response wasn't valid JSON. Showing raw output instead.")
        parsed = {"recommendations": [], "overall_note": text}

    return parsed


# ---------------------------------------------------------------------------
# Hugging Face: dataset chatbot (with conversation history)
# ---------------------------------------------------------------------------

# Questions like "who sings X" / "singer of X" / "song by X" are lookups: if
# the title isn't found we must say so instead of letting the model guess.
LOOKUP_PATTERN = re.compile(
    r"\b(who\s+(sings|sang|sing|wrote|made|is\s+the\s+(singer|artist))|"
    r"(singer|artist|performer)\s+of|sung\s+by|which\s+artist)\b",
    re.IGNORECASE,
)

# Single words too generic to count as a track-title match on their own.
_GENERIC_WORDS = {
    "the", "a", "an", "is", "are", "was", "who", "what", "when", "where",
    "why", "how", "of", "for", "and", "or", "to", "in", "on", "at", "me",
    "my", "you", "your", "i", "it", "this", "that", "song", "songs",
    "track", "tracks", "artist", "singer", "sings", "sang", "music",
    "best", "top", "more", "some", "any", "give", "tell", "about",
}


def find_title_matches(df, question: str, max_rows: int = 15):
    """Rows whose track_name appears in the question, e.g. "save your tears".
    Builds every 1-8 word phrase from the question and matches them against
    lowercased track names with pandas (vectorised, no model involved).
    Single generic words ("song", "best") are ignored."""
    if "track_name" not in df.columns:
        return df.iloc[0:0]
    words = re.findall(r"[a-z0-9']+", question.lower())
    phrases = set()
    for n in range(1, 9):
        for i in range(len(words) - n + 1):
            chunk = words[i:i + n]
            if n == 1 and (chunk[0] in _GENERIC_WORDS or len(chunk[0]) < 3):
                continue
            if all(w in _GENERIC_WORDS for w in chunk):
                continue
            phrases.add(" ".join(chunk))
    if not phrases:
        return df.iloc[0:0]
    normalized = df["track_name"].str.lower().str.replace(r"[^a-z0-9' ]+", " ", regex=True) \
        .str.replace(r"\s+", " ", regex=True).str.strip()
    hits = df[normalized.isin(phrases)]
    if hits.empty:
        return hits
    # Prefer the longest matching title (e.g. "save your tears" over "tears").
    longest = normalized[hits.index].str.split().str.len().max()
    hits = hits[normalized[hits.index].str.split().str.len() == longest]
    return hits.sort_values("popularity", ascending=False).head(max_rows)


def build_chat_context(df, question: str, history=None, max_rows: int = 80) -> str:
    """Builds a small, relevant slice of the dataset as grounding context
    for the Hugging Face model, so it answers from real rows instead of
    hallucinating tracks or artists. Also looks at the last couple of user
    messages, so a follow-up like "more of those" still pulls the right rows."""
    recent_user = [c for r, c in (history or []) if r == "user"][-2:]
    words = set(re.findall(r"[a-z0-9']+", " ".join(recent_user + [question]).lower()))

    scored = df.copy()
    scored["_hit"] = 0
    if "genre" in scored.columns:
        scored["_hit"] += scored["genre"].str.lower().isin(words).astype(int) * 2
    if "mood" in scored.columns:
        scored["_hit"] += scored["mood"].str.lower().apply(
            lambda m: any(w in words for w in re.findall(r"[a-z0-9']+", m))
        ).astype(int) * 2
    if "artist_name" in scored.columns:
        scored["_hit"] += scored["artist_name"].str.lower().apply(
            lambda a: any(w in words for w in re.findall(r"[a-z0-9']+", a))
        ).astype(int) * 3

    cols = [c for c in ["track_name", "artist_name", "genre", "mood", "popularity"] if c in scored.columns]

    # Rows whose track title is named in the question come first, so a
    # lookup like "who sings save your tears" is answered from the real row.
    title_hits = find_title_matches(df, question)
    top = scored.sort_values(["_hit", "popularity"], ascending=[False, False]).head(max_rows)
    if not title_hits.empty:
        top = top[~top.index.isin(title_hits.index)]

    parts = []
    if not title_hits.empty:
        parts.append(
            "Rows whose track title appears in the user's question "
            "(use these to answer; the artist_name here is the correct artist):\n"
            + json.dumps(title_hits[cols].to_dict(orient="records"), ensure_ascii=False)
        )
    elif LOOKUP_PATTERN.search(question):
        parts.append(
            "NOTE: the song the user asked about was NOT found in the dataset. "
            "Tell them it isn't in the dataset. Do NOT guess an artist."
        )
    parts.append(
        "Sample rows (JSON):\n"
        + json.dumps(top[cols].head(max_rows).to_dict(orient="records"), ensure_ascii=False)
    )
    return "\n\n".join(parts)


def build_dataset_stats(df) -> str:
    """Whole-dataset facts computed by pandas, so the model never has to
    count or average rows from the small excerpt it is shown."""
    stats = {"total_tracks": int(len(df))}
    if "artist_name" in df.columns:
        stats["unique_artists"] = int(df["artist_name"].nunique())
    if "genre" in df.columns:
        stats["unique_genres"] = int(df["genre"].nunique())
        stats["genres"] = sorted(df["genre"].dropna().unique().tolist())
    if "mood" in df.columns:
        stats["tracks_per_mood"] = {k: int(v) for k, v in df["mood"].value_counts().items()}
    for col in ["popularity", "danceability", "energy", "valence", "tempo"]:
        if col in df.columns:
            stats[f"average_{col}"] = round(float(df[col].mean()), 3)
    return json.dumps(stats, ensure_ascii=False)


def _build_chat_messages(df, question: str, history=None):
    """System prompt (with dataset excerpt) + the last few turns of the
    conversation + the new question."""
    context = build_chat_context(df, question, history)
    system_prompt = (
        "You are a music dataset assistant. Answer ONLY using the track "
        "data given below. Any track or artist you mention must come from "
        "this data, never invented. You have no knowledge of music outside "
        "this dataset; if the data can't answer the question, say so "
        "plainly instead of guessing. Use the earlier messages in this "
        "conversation to understand what the user means by words like "
        "'those', 'that artist', or 'more'. When the user asks for a list, "
        "give as many items as they ask for (or as many as are useful) and "
        "finish the list completely. When asked who sings/made a song, answer "
        "ONLY from the artist_name of the matching row; if no row matches, "
        "say the song isn't in the dataset. For any question about totals, counts, "
        "or averages across the whole dataset, use ONLY the 'Whole-dataset "
        "statistics' below; the excerpt is just a small sample and must "
        "never be counted.\n\nWhole-dataset statistics (JSON):\n"
        + build_dataset_stats(df)
        + "\n\nDataset context:\n" + context
    )
    messages = [{"role": "system", "content": system_prompt}]
    for role, content in (history or [])[-HISTORY_TURNS:]:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def ask_dataset_question(hf_model: str, df, question: str, history=None, *args, **kwargs) -> str:
    """Non-streaming version. Caller is expected to have already checked
    relevance. `history` is a list of (role, content) tuples."""
    from huggingface_hub import InferenceClient

    hf_token = get_hf_token()
    if not hf_token:
        raise RuntimeError(
            "No Hugging Face token found. Add HF_TOKEN to "
            ".streamlit/secrets.toml (see secrets.toml.example)."
        )

    client = InferenceClient(model=hf_model, token=hf_token)
    response = client.chat_completion(
        messages=_build_chat_messages(df, question, history),
        max_tokens=CHAT_MAX_TOKENS,
        temperature=0.6,
    )
    return response.choices[0].message.content.strip()


def ask_dataset_question_stream(hf_model: str, df, question: str, history=None, *args, **kwargs):
    """Streaming version: yields the answer chunk by chunk, meant to be
    passed straight to `st.write_stream()`. `history` is a list of
    (role, content) tuples from earlier in the conversation."""
    from huggingface_hub import InferenceClient

    hf_token = get_hf_token()
    if not hf_token:
        raise RuntimeError(
            "No Hugging Face token found. Add HF_TOKEN to "
            ".streamlit/secrets.toml (see secrets.toml.example)."
        )

    client = InferenceClient(model=hf_model, token=hf_token)
    stream = client.chat_completion(
        messages=_build_chat_messages(df, question, history),
        max_tokens=CHAT_MAX_TOKENS,
        temperature=0.6,
        stream=True,
    )
    for chunk in stream:
        # The final chunk of a stream can have an empty `choices` list;
        # indexing it caused "list index out of range".
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
