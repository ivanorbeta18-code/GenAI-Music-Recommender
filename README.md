# GenAI Music Recommender (MVP)

Recommends tracks based on your favorite **genre**, **artist**, and **mood**,
using a real Spotify tracks dataset (~114k tracks). Pandas narrows the
dataset to a relevant shortlist; Claude (Anthropic API) reasons over that
shortlist to pick the final tracks and explain each pick. A second panel
lets you ask a Hugging Face-powered chatbot free-form questions about the
dataset.

## Why "mood" instead of "time period"

The bundled dataset (`data/spotify_dataset.csv`) doesn't include a release
year. Instead, each track is scored on two audio features it *does* have:

- **valence** — how musically positive/happy a track sounds
- **energy** — how intense/active a track sounds

Those two combine into four mood quadrants used throughout the app:

| valence | energy | mood              |
|---------|--------|-------------------|
| high    | high   | Happy & Energetic |
| high    | low    | Chill & Content   |
| low     | high   | Angry & Intense   |
| low     | low    | Sad & Mellow      |

**Popularity** (0–100) is the second filter axis, letting users dial between
mainstream hits and obscure/unheard tracks.

## Project structure

```
music_recommender_app/
├── app.py                          # Streamlit UI + app flow (2 tabs)
├── requirements.txt
├── data/
│   └── spotify_dataset.csv         # real Spotify tracks dataset (~114k rows)
├── utils/
│   ├── data_utils.py               # load/clean/filter + mood scoring + chatbot relevance gate
│   └── ai_utils.py                 # Claude (recommendations) + Hugging Face (chatbot)
└── .streamlit/
    └── secrets.toml.example        # template for your API keys
```

## 1. Dataset

`data/spotify_dataset.csv` needs at least these columns (the app
auto-renames a few common variants, e.g. Kaggle's `artists`/`track_genre`):

| required    | notes            |
|-------------|------------------|
| track_name  | song title       |
| artist_name | artist           |
| genre       | genre label      |

Optional columns like `popularity`, `valence`, `energy`, `danceability`,
`tempo`, etc. are used when present (they power the mood scoring and the
popularity filter) but aren't strictly required. You can swap in your own
CSV via the app's sidebar — no code changes needed.

## 2. Run locally

```bash
cd music_recommender_app
pip install -r requirements.txt

# Option A: paste your keys in the sidebar at runtime (fastest for testing)
streamlit run app.py

# Option B: use a secrets file instead
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and add your real keys
streamlit run app.py
```

- Get a Claude API key at https://console.anthropic.com.
- Get a Hugging Face access token at https://huggingface.co/settings/tokens.

## 3. How it works

### Recommendation engine (Tab 1)
1. **Load & clean** — `utils/data_utils.load_dataset()` loads the CSV with
   Pandas, normalizes column names, drops bad rows, and computes a `mood`
   column from `valence` + `energy`.
2. **Pick favorites** — multiselect genres, multiselect artists (scoped to
   chosen genres), multiselect moods, and a popularity-range slider.
3. **Narrow candidates** — `filter_candidates()` scores every track by how
   well it matches genre/artist/mood/popularity and keeps the top ~60, so
   the AI step only ever sees a relevant, small shortlist.
4. **GenAI reasoning** — `utils/ai_utils.get_recommendations()` sends that
   shortlist plus your stated favorites to Claude, which is instructed to
   only pick from the list (no invented tracks) and explain each pick.
5. **Display & visualize** — recommendations are shown as cards with a
   one-line reason each, plus two bar charts (genre mix, mood mix) built
   with Streamlit's native charting.

### Dataset chatbot (Tab 2)
1. You type a free-form question in a `st.chat_input` box.
2. **Relevance gate** — `data_utils.is_dataset_related()` checks the
   question's words against a vocabulary built from the dataset's genres,
   artist-name words, and a set of music/recommendation keywords (e.g.
   *recommend*, *mood*, *popular*, *unheard*). Off-topic input (`"hi"`,
   `"who is the US president"`) is declined immediately, **without**
   calling any AI model.
3. **Grounded answer** — for on-topic questions, `ai_utils.ask_dataset_question()`
   pulls a small relevant slice of the dataset (matching genre/mood/artist
   words in the question) as context, then sends it to a Hugging Face chat
   model with instructions to only reference tracks/artists from that
   context.

## 4. Deploy to Streamlit Community Cloud

1. Push this folder to a public (or connected private) GitHub repo.
2. Go to https://share.streamlit.io and click **New app**.
3. Point it at your repo, branch, and `app.py`.
4. In the app's **Settings → Secrets**, paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   HF_API_KEY = "hf_..."
   ```
5. Deploy. Users won't need to paste API keys — the app already falls back
   to `st.secrets` when present.

## 5. Next steps

- Swap `DEFAULT_HF_MODEL` in `utils/ai_utils.py` for a different
  instruction-tuned model if your Hugging Face token doesn't have access
  to the default one.
- Tighten the chatbot's relevance gate further (e.g. a small zero-shot
  classification call) if keyword matching lets too much through.
- Add more filters (danceability threshold, explicit-content toggle).
- Swap the bar charts for Plotly/Altair for richer, interactive visuals.
- Cache Claude/Hugging Face responses per unique query to reduce API calls.