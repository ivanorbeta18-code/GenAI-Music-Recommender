# GenAI Music Recommender (MVP)

CS 315 – Activity 3

Recommends tracks based on your favorite **genre**, **artist**, and **mood**, using a real Spotify tracks dataset (~114k tracks). Pandas narrows the dataset to a relevant shortlist, then a **Hugging Face** model reasons over that shortlist to pick the final tracks and explain each pick. A second tab is a chatbot, powered by the same model, that answers free-form questions about the dataset.

> **Scope:** every recommendation and chatbot answer is limited to the tracks in the loaded dataset. The app is not connected to Spotify or any live music service.

## Why "mood" instead of "time period"

The dataset has no release year. Instead, each track is scored on two audio features it does have:

- **valence**: how musically positive a track sounds
- **energy**: how intense or active a track sounds

Splitting each at 0.5 gives four mood quadrants used throughout the app:

| valence | energy | mood              |
|---------|--------|-------------------|
| high    | high   | Happy & Energetic |
| high    | low    | Chill & Content   |
| low     | high   | Angry & Intense   |
| low     | low    | Sad & Mellow      |

**Popularity** (0-100) is the second filter, letting users choose between obscure and mainstream tracks.

## Project structure

```
music_recommender_app/
├── app.py                      # Streamlit UI and app flow (2 tabs)
├── requirements.txt
├── README.md
├── .gitignore                  # keeps .streamlit/secrets.toml out of Git
├── data/
│   └── spotify_dataset.csv     # Spotify tracks dataset (~114k rows)
├── utils/
│   ├── __init__.py
│   ├── data_utils.py           # load/clean, mood scoring, candidate filter, chatbot relevance gate
│   ├── ai_utils.py             # Hugging Face calls (recommendations + chatbot)
│   └── theme.py                # CSS, mood colors, badges, valence/energy diagram
└── .streamlit/
    ├── config.toml             # dark theme
    ├── secrets.toml            # your real HF_TOKEN (git-ignored)
    └── secrets.toml.example    # template for the secrets file
```

## 1. Dataset

`data/spotify_dataset.csv` needs at least these columns. The app auto-renames common Kaggle variants (`artists`/`artist` to `artist_name`, `track_genre`/`genres` to `genre`) and drops a stray unnamed index column.

| required    | notes       |
|-------------|-------------|
| track_name  | song title  |
| artist_name | artist      |
| genre       | genre label |

Optional numeric columns (`popularity`, `valence`, `energy`, `danceability`, `tempo`, etc.) are used when present. `valence` and `energy` power the mood scoring, and `popularity` powers the popularity filter. You can swap in your own Spotify-style CSV from the sidebar with no code changes.

## 2. Setup and run locally

```bash
cd music_recommender_app
pip install -r requirements.txt
```

1. Create a Hugging Face access token at https://huggingface.co/settings/tokens.
2. Copy the template and add your token:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   ```toml
   # .streamlit/secrets.toml
   HF_TOKEN = "hf_..."
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

There are no key fields in the UI. The token is read from `st.secrets`. **Never commit `secrets.toml`**; `.gitignore` already excludes it. If a token is ever exposed, revoke it in your Hugging Face settings and create a new one.

## 3. How it works

### Recommendation engine (Tab 1: Get Recommendations)
1. **Load and clean**: `data_utils.load_dataset()` reads the CSV, normalizes column names, drops rows missing required fields, coerces numeric columns, fills missing valence/energy with the median, clips popularity to 0-100, adds the `mood` column, and drops duplicate (track, artist) pairs. The result is cached with `@st.cache_data`.
2. **Pick favorites**: multiselect genres, artists (scoped to the chosen genres) and moods, plus a popularity range slider and a "how many recommendations" slider (3-15). If nothing is selected, a confirmation dialog warns that results will be based on popularity alone.
3. **Narrow candidates**: `filter_candidates()` scores every track (genre match +2, artist match +3, mood match +2, popularity in range +1, plus a small popularity bonus) and keeps the top 60, so the model only sees a small, relevant shortlist.
4. **GenAI reasoning**: `ai_utils.get_recommendations()` sends the shortlist and the user's favorites to the model, which is told to pick only from the list and to return JSON with `track_name`, `artist_name`, `genre`, `mood` and a one-sentence `reason`. Code fences are stripped, and if the JSON is invalid the app shows the raw output instead of crashing.
5. **Display and visualize**: results appear as cards with a left border colored by mood, plus mood and genre badges. Two Altair bar charts show the genre mix and mood mix, and an expander shows the full pandas candidate shortlist.

### Dataset chatbot (Tab 2: Ask the Dataset)
1. You type a question into `st.chat_input`.
2. **Relevance gate**: `data_utils.is_dataset_related()` checks the question against a vocabulary of music keywords, genre words and full multi-word artist names. Off-topic input (e.g. "hi", "who is the US president") gets a canned refusal **without calling the model**.
3. **Grounded answer**: for on-topic questions, `ai_utils.ask_dataset_question_stream()` pulls the 40 most relevant rows as context and asks the model to answer only from them. The reply streams into the chat with `st.write_stream`.
4. The chat lives in a fixed-height scroll container with a **Clear chat** button.

### Model
Both features use one instruction-tuned model through the Hugging Face Inference API: `Qwen/Qwen3-4B-Instruct-2507` (set by `DEFAULT_HF_MODEL` in `utils/ai_utils.py`).

## 4. Deploy to Streamlit Community Cloud

1. Push the project folder to GitHub. `secrets.toml` stays out because of `.gitignore`.
2. Go to https://share.streamlit.io and click **New app**.
3. Select your repo, branch and `app.py`.
4. In the app's **Settings → Secrets**, paste:
   ```toml
   HF_TOKEN = "hf_..."
   ```
5. Deploy. Users never see or type a key.

## 5. Testing notes

Tested with on-topic chat prompts ("Recommend me a chill indie artist"), off-topic prompts ("hi"), each filter alone and combined, no favorites selected, narrow popularity ranges (0-20, 80-100), and a custom uploaded CSV.

Improvements from testing: artist names are matched as full phrases (single words like "weather" made almost anything look on-topic), chat moved into a scrollable box, replies stream instead of waiting on a spinner, and a JSON-fallback parser was added.

## 6. Next steps

- Add more filters (danceability, tempo, explicit-content toggle).
- Add a "similar to this track" feature.
- Cache model responses per unique query to reduce API calls.
- Replace the keyword relevance gate with a small zero-shot classifier if too much off-topic input slips through.
- Try a different instruction-tuned model if your token lacks access to the default (for example `HuggingFaceTB/SmolLM2-1.7B-Instruct` for faster, lighter responses).
