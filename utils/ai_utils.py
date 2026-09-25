"""
GenAI layer — a single Hugging Face instruction-tuned model powers both
features of the app:

- **Recommendation reasoning** — given a pandas-narrowed candidate
  shortlist, the model picks + explains the final track recommendations.
- **Dataset chatbot** — answers free-form user questions about the
  dataset (e.g. "recommend me a chill artist", "give me a random unheard
  pop track"). The app only calls it for questions that pass
  `data_utils.is_dataset_related()` — everything else (greetings, trivia,
  general knowledge) gets a canned refusal instead of an API call.

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
# Hugging Face: dataset chatbot
# ---------------------------------------------------------------------------

def build_chat_context(df, question: str, max_rows: int = 40) -> str:
    """Builds a small, relevant slice of the dataset as grounding context
    for the Hugging Face model, so it answers from real rows instead of
    hallucinating tracks or artists."""
    words = set(re.findall(r"[a-z0-9']+", question.lower()))

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

    top = scored.sort_values(["_hit", "popularity"], ascending=[False, False]).head(max_rows)
    cols = [c for c in ["track_name", "artist_name", "genre", "mood", "popularity"] if c in top.columns]
    records = top[cols].to_dict(orient="records")
    return json.dumps(records, ensure_ascii=False)


def ask_dataset_question(hf_model: str, df, question: str) -> str:
    """Answers a dataset-related question using a Hugging Face chat model.
    Caller is expected to have already checked `is_dataset_related()` —
    this function does not re-check relevance, it just answers. Uses the
    token embedded via `get_hf_token()`."""
    from huggingface_hub import InferenceClient

    hf_token = get_hf_token()
    if not hf_token:
        raise RuntimeError(
            "No Hugging Face token found. Add HF_TOKEN to "
            ".streamlit/secrets.toml (see secrets.toml.example)."
        )

    context = build_chat_context(df, question)
    system_prompt = (
        "You are a music dataset assistant. Answer ONLY using the track "
        "data given to you below — if you recommend or mention a track or "
        "artist, it must come from this data, never invented. You have no "
        "knowledge of music outside this dataset; if the data below can't "
        "answer the question, say so plainly instead of guessing. Keep "
        "answers short (2-4 sentences).\n\nDataset excerpt (JSON):\n" + context
    )

    client = InferenceClient(model=hf_model, token=hf_token)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    response = client.chat_completion(messages=messages, max_tokens=300, temperature=0.6)
    return response.choices[0].message.content.strip()


def ask_dataset_question_stream(hf_model: str, df, question: str):
    """Same as `ask_dataset_question`, but yields the answer as it's
    generated (token by token / chunk by chunk) instead of returning it
    all at once. Meant to be passed straight to `st.write_stream()` so the
    chat UI shows the model "typing" and the page follows it down, rather
    than sitting on a spinner and then dumping the full answer at once.
    Caller is expected to have already checked `is_dataset_related()`."""
    from huggingface_hub import InferenceClient

    hf_token = get_hf_token()
    if not hf_token:
        raise RuntimeError(
            "No Hugging Face token found. Add HF_TOKEN to "
            ".streamlit/secrets.toml (see secrets.toml.example)."
        )

    context = build_chat_context(df, question)
    system_prompt = (
        "You are a music dataset assistant. Answer ONLY using the track "
        "data given to you below — if you recommend or mention a track or "
        "artist, it must come from this data, never invented. You have no "
        "knowledge of music outside this dataset; if the data below can't "
        "answer the question, say so plainly instead of guessing. Keep "
        "answers short (2-4 sentences).\n\nDataset excerpt (JSON):\n" + context
    )

    client = InferenceClient(model=hf_model, token=hf_token)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    stream = client.chat_completion(messages=messages, max_tokens=300, temperature=0.6, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta