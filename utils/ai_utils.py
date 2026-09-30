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
CHAT_MAX_TOKENS = 800

# Hard ceiling on how many items the chatbot may list in one answer.
MAX_LIST_ITEMS = 10

# Rows of dataset context sent to the model. Fewer rows = less temptation
# to enumerate everything.
CHAT_CONTEXT_ROWS = 40

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

# Shown (without calling the model) when the user asks for every song.
LIST_ALL_WARNING = (
    "Listing every song isn't viable. This dataset has about {total:,} tracks, "
    "which would be an extremely long list, so I can show **at most 10** at a time.\n\n"
    "Are you sure you want the top 10? Reply **yes** to continue, or narrow it "
    "down (a genre, artist, or mood) so the 10 I pick fit what you want."
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

def build_chat_context(df, question: str, history=None, max_rows: int = CHAT_CONTEXT_ROWS) -> str:
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

    top = scored.sort_values(["_hit", "popularity"], ascending=[False, False]).head(max_rows)
    cols = [c for c in ["track_name", "artist_name", "genre", "mood", "popularity"] if c in top.columns]
    records = top[cols].to_dict(orient="records")
    return json.dumps(records, ensure_ascii=False)


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
        "'those', 'that artist', or 'more'. "
        f"LIST LIMIT: never list more than {MAX_LIST_ITEMS} items in a "
        "single answer, even if the data contains more. If the user asks "
        f"for more than {MAX_LIST_ITEMS}, or for 'all' of something, give "
        f"the best {MAX_LIST_ITEMS} and end with one short sentence "
        "offering to show more. If they ask for a specific number "
        f"{MAX_LIST_ITEMS} or lower, give exactly that many. Finish the "
        "list cleanly; don't cut off mid-item. "
        "CLARIFY WHEN UNSURE: if the request is vague or ambiguous (for "
        "example no genre, artist, or mood is given, or it's unclear what "
        "'those' refers to), do not guess. Ask ONE short clarifying "
        "question instead of answering.\n\n"
        "Dataset excerpt (JSON):\n" + context
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
