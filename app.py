"""
GenAI Music Recommender — Streamlit MVP
CS 315 — Activity 3

Recommends tracks based on a user's favorite genre(s), favorite artist(s),
and mood, using a real Spotify tracks dataset. The dataset has no
release-year column, so **mood** (derived from valence + energy) and
**popularity** stand in for "time period" as the narrowing filters. Pandas
narrows the dataset down to a relevant shortlist, then a Hugging Face
model (GenAI) reasons over that shortlist to pick and explain the final
recommendations.

A second GenAI feature — the same Hugging Face model, used as a chatbot —
answers free-form questions about the dataset ("recommend me a chill
artist", "give me a random unheard pop track") and politely declines
anything unrelated to the dataset ("hi", "who is the US president").

No API key is ever typed into the UI: the Hugging Face token is embedded
in the app via `.streamlit/secrets.toml` (see `secrets.toml.example` and
`.gitignore`, which keeps the real secrets file out of version control).
"""

import os

import altair as alt
import pandas as pd
import streamlit as st

from utils.ai_utils import (
    DEFAULT_HF_MODEL,
    REFUSAL_MESSAGE,
    ask_dataset_question_stream,
    get_hf_token,
    get_recommendations,
)
from utils.data_utils import build_vocab, filter_candidates, is_dataset_related, load_dataset
from utils.theme import (
    MOOD_COLORS,
    genre_badge,
    inject_css,
    mood_badge,
    mood_stripe_css,
    quadrant_diagram,
)

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "spotify_dataset.csv")

st.set_page_config(page_title="GenAI Music Recommender", page_icon="🎧", layout="wide")
st.markdown(inject_css(), unsafe_allow_html=True)

st.markdown(
    '<div class="app-hero">' + quadrant_diagram(size=26, labels=False)
    + '<h1 class="app-hero__title">GenAI Music Recommender</h1></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Tell it what you love — genre, artist, mood — and GenAI picks tracks "
    "for you from the dataset. Or ask the chatbot anything about the data."
)
st.warning(
    "⚠️ **Scope disclosure:** every recommendation and every chatbot answer "
    "in this app is limited strictly to tracks/artists/genres in the "
    "dataset currently loaded (bundled dataset, or your uploaded CSV)" 
    "and it is uploaded in 2022. It cannot recommend or discuss music "
    "outside that dataset, and it isn't "
    "connected to Spotify or any other live music service.",
    icon="⚠️",
)

hf_token = get_hf_token()
hf_model = DEFAULT_HF_MODEL
if not hf_token:
    st.error(
        "No Hugging Face token found. Add `HF_TOKEN` to "
        "`.streamlit/secrets.toml` (copy `secrets.toml.example` and fill "
        "in your real token — that file is git-ignored, so it's safe to "
        "keep it there). Nothing else in this app needs a key.",
        icon="🔑",
    )

# ---------------------------------------------------------------------------
# Sidebar: dataset + (optional) model choice — no key inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Dataset")
    uploaded = st.file_uploader(
        "Upload a Spotify-style CSV (optional)",
        type=["csv"],
        help="Needs at least: track_name, artists (or artist_name), "
             "track_genre (or genre). If you skip this, the bundled "
             "dataset is used.",
    )

    st.caption("No API key needed here — it's embedded in the app's config.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    df = load_dataset(uploaded if uploaded is not None else DEFAULT_DATA_PATH)
except ValueError as e:
    st.error(str(e))
    st.stop()

if uploaded is None:
    st.info(
        "Using the bundled Spotify Tracks dataset (~114k tracks). Upload "
        "your own Spotify-style CSV in the sidebar to use different data. "
        "**Note:** this dataset has no release-year column, so mood "
        "(valence + energy) and popularity are used instead of a "
        "time-period filter.",
        icon="ℹ️",
    )

vocab = build_vocab(df)

tab_recommend, tab_chat = st.tabs(["🎯 Get Recommendations", "💬 Ask the Dataset"])

# ---------------------------------------------------------------------------
# Tab 1: Recommendations (pandas shortlist -> Hugging Face picks + explains)
# ---------------------------------------------------------------------------
with tab_recommend:
    st.subheader("Your favorites")
    col1, col2, col3 = st.columns(3)

    with col1:
        all_genres = sorted(df["genre"].dropna().unique().tolist())
        fav_genres = st.multiselect("Favorite genre(s)", all_genres)

    with col2:
        if fav_genres:
            artist_pool = sorted(df[df["genre"].isin(fav_genres)]["artist_name"].unique().tolist())
        else:
            artist_pool = sorted(df["artist_name"].unique().tolist())
        fav_artists = st.multiselect("Favorite artist(s)", artist_pool)

    with col3:
        all_moods = sorted(df["mood"].dropna().unique().tolist())
        fav_moods = st.multiselect(
            "Favorite mood(s)",
            all_moods,
            help="Derived from each track's valence (musical positivity) and energy — "
                 "stands in for 'time period' since this dataset has no release year.",
        )
        diagram_col, caption_col = st.columns([1, 2])
        with diagram_col:
            st.markdown(quadrant_diagram(fav_moods, size=64), unsafe_allow_html=True)
        with caption_col:
            st.markdown(
                '<p class="quadrant-caption">Where your mood picks sit on the '
                "valence/energy plane this dataset measures.</p>",
                unsafe_allow_html=True,
            )

    popularity_range = st.slider(
        "Popularity range you're open to (0 = obscure, 100 = mainstream)", 0, 100, (0, 100)
    )
    n_recommend = st.slider("How many recommendations?", 3, 15, 8)

    go = st.button("🎯 Get recommendations", type="primary")

    no_favorites = not fav_genres and not fav_artists and not fav_moods

    @st.dialog("Get recommendations without favorites?")
    def confirm_no_favorites_dialog():
        st.write(
            "You haven't picked any favorite genres, artists, or moods — "
            "recommendations will be based on popularity alone and won't "
            "really reflect your taste."
        )
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button("Yes, continue", type="primary", use_container_width=True):
                st.session_state["needs_favorites_confirmation"] = False
                st.session_state["run_recs_confirmed"] = True
                st.rerun()
        with cancel_col:
            if st.button("Cancel", use_container_width=True):
                st.session_state["needs_favorites_confirmation"] = False
                st.rerun()

    run_recs = False

    if go:
        if no_favorites:
            st.session_state["needs_favorites_confirmation"] = True
        else:
            run_recs = True

    if st.session_state.get("needs_favorites_confirmation"):
        confirm_no_favorites_dialog()

    if st.session_state.pop("run_recs_confirmed", False):
        run_recs = True

    if run_recs:
        if not hf_token:
            st.warning("Add HF_TOKEN to .streamlit/secrets.toml first.")
            st.stop()

        with st.spinner("Narrowing down candidates from the dataset..."):
            candidates = filter_candidates(df, fav_genres, fav_artists, fav_moods, popularity_range)

        with st.spinner("Asking the model to reason about your taste and pick tracks..."):
            try:
                result = get_recommendations(
                    hf_model, fav_genres, fav_artists, fav_moods, popularity_range, candidates, n_recommend
                )
            except Exception as e:
                st.error(f"Couldn't reach the Hugging Face model: {e}")
                st.stop()

        recs = result.get("recommendations", [])
        note = result.get("overall_note", "")

        if note:
            st.success(note)

        if not recs:
            st.warning("No recommendations came back. Try widening your filters.")
        else:
            rec_df = pd.DataFrame(recs)

            st.subheader("Recommended tracks")
            for i, row in rec_df.reset_index(drop=True).iterrows():
                st.markdown(mood_stripe_css(i, row.get("mood", "")), unsafe_allow_html=True)
                with st.container(border=True, key=f"rec-card-{i}"):
                    st.markdown('<div class="rec-card-marker"></div>', unsafe_allow_html=True)
                    st.markdown(
                        f"**{row.get('track_name', 'Unknown')}** — {row.get('artist_name', 'Unknown')}"
                    )
                    badges = []
                    if row.get("mood"):
                        badges.append(mood_badge(str(row["mood"])))
                    if row.get("genre"):
                        badges.append(genre_badge(str(row["genre"])))
                    if badges:
                        st.markdown(" ".join(badges), unsafe_allow_html=True)
                    if row.get("reason"):
                        st.write(row["reason"])

            st.subheader("Visualize the recommendations")
            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                if "genre" in rec_df.columns:
                    st.caption("Genre mix of your recommendations")
                    genre_counts = rec_df["genre"].value_counts().reset_index()
                    genre_counts.columns = ["genre", "count"]
                    genre_chart = (
                        alt.Chart(genre_counts)
                        .mark_bar(color=MOOD_COLORS["Happy & Energetic"], cornerRadiusEnd=3)
                        .encode(
                            x=alt.X("count:Q", title=None),
                            y=alt.Y("genre:N", sort="-x", title=None),
                        )
                        .properties(height=160)
                        .configure_view(strokeWidth=0)
                        .configure_axis(grid=False, labelFontSize=10)
                    )
                    st.altair_chart(genre_chart, use_container_width=True)

            with viz_col2:
                if "mood" in rec_df.columns:
                    st.caption("Mood mix of your recommendations")
                    mood_counts = rec_df["mood"].value_counts().reset_index()
                    mood_counts.columns = ["mood", "count"]
                    mood_chart = (
                        alt.Chart(mood_counts)
                        .mark_bar(cornerRadiusEnd=3)
                        .encode(
                            x=alt.X("count:Q", title=None),
                            y=alt.Y("mood:N", sort="-x", title=None),
                            color=alt.Color(
                                "mood:N",
                                scale=alt.Scale(
                                    domain=list(MOOD_COLORS.keys()),
                                    range=list(MOOD_COLORS.values()),
                                ),
                                legend=None,
                            ),
                        )
                        .properties(height=160)
                        .configure_view(strokeWidth=0)
                        .configure_axis(grid=False, labelFontSize=10)
                    )
                    st.altair_chart(mood_chart, use_container_width=True)

            with st.expander("See the full candidate shortlist pandas selected"):
                st.dataframe(candidates, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Dataset chatbot (Hugging Face, gated to on-topic questions)
# ---------------------------------------------------------------------------
with tab_chat:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    header_col, clear_col = st.columns([5, 1])
    with header_col:
        st.subheader("Ask the dataset")
        st.caption(
            "Powered by the same Hugging Face model, grounded in this dataset. "
            "It'll only answer dataset-related questions — recommend an "
            "artist, surface a random unheard track, describe a mood — and "
            "will politely decline anything else (small talk, general trivia, "
            "etc.). **It only knows about tracks in the loaded dataset** — it "
            "cannot answer about songs, artists, or releases outside it."
        )
    with clear_col:
        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    # Fixed header above, bounded scrollable message pane below — like a
    # messenger window. New messages push older ones up within this box
    # instead of growing the whole page.
    chat_box = st.container(height=320, border=True)
    with chat_box:
        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(content)

    question = st.chat_input('e.g. "Recommend me a chill indie artist"')

    if question:
        st.session_state.chat_history.append(("user", question))
        with chat_box:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                if not is_dataset_related(question, vocab):
                    answer = REFUSAL_MESSAGE
                    st.write(answer)
                elif not hf_token:
                    answer = "Add HF_TOKEN to .streamlit/secrets.toml first."
                    st.warning(answer)
                else:
                    try:
                        answer = st.write_stream(ask_dataset_question_stream(hf_model, df, question))
                    except Exception as e:
                        answer = f"Couldn't reach the Hugging Face model: {e}"
                        st.error(answer)

        st.session_state.chat_history.append(("assistant", answer))

st.divider()
st.caption(
    "MVP built with Streamlit + Pandas + Hugging Face (recommendations "
    "and chatbot, one model doing both jobs). Deploy via Streamlit "
    "Community Cloud; set HF_TOKEN in the app's Settings → Secrets "
    "instead of a local secrets.toml for a shared deployment."
)