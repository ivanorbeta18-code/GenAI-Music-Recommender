# 🎧 GenAI Music Recommender (MVP)

**CS 315 – Application Development and Emerging Technologies · Activity 3**

A Streamlit prototype that recommends music from a real Spotify dataset (~114,000 tracks) based on a user's favorite **genres**, **artists** and **moods**. Pandas narrows the dataset to a relevant shortlist, then a Hugging Face language model picks the final tracks and explains each choice. A second tab lets users chat with the dataset.

> **Scope disclosure:** every recommendation and chatbot answer is limited to the tracks in the loaded dataset (the bundled one or an uploaded CSV). The app is not connected to Spotify or any live music service and cannot discuss music outside the data.

---

## Table of contents

1. [Features](#1-features)
2. [Model and tech stack](#2-model-and-tech-stack)
3. [Design decisions: the whys](#3-design-decisions-the-whys)
4. [How it works](#4-how-it-works)
5. [Project structure](#5-project-structure)
6. [Dataset](#6-dataset)
7. [Setup and run locally](#7-setup-and-run-locally)
8. [Using the app](#8-using-the-app)
9. [Deploy to Streamlit Community Cloud](#9-deploy-to-streamlit-community-cloud)
10. [Security notes](#10-security-notes)
11. [Testing and iteration](#11-testing-and-iteration)
12. [Known limitations](#12-known-limitations)
13. [Next steps](#13-next-steps)
14. [Activity 3 checklist](#14-activity-3-checklist)

---

## 1. Features

### Recommendations (Tab 1: 🎯 Get Recommendations)
- **Favorite genre(s):** multiselect over every genre in the dataset.
- **Favorite artist(s):** multiselect that is **scoped to the chosen genres**, so the list stays manageable.
- **Favorite mood(s):** four moods derived from audio features (Happy & Energetic, Chill & Content, Angry & Intense, Sad & Mellow).
- **Valence/energy diagram:** a small SVG that highlights where the selected moods sit on the plane.
- **Popularity range slider:** 0 (obscure) to 100 (mainstream).
- **Recommendation count slider:** 3 to 15 tracks.
- **Confirmation dialog** when the user clicks the button with no favorites selected, warning that results will lean on popularity alone.
- **AI-written reasons:** each recommended track comes with a one-sentence explanation tied to the user's stated preferences, plus an overall note summarizing the detected taste profile.
- **Recommendation cards** with a left border colored by mood, plus mood and genre badges.
- **Visualizations:** two Altair bar charts (genre mix and mood mix of the results).
- **Transparency expander:** shows the full candidate shortlist that pandas selected, so users can see what the AI had to choose from.

### Dataset chatbot (Tab 2: 💬 Ask the Dataset)
- Free-form questions such as "recommend me a chill indie artist" or "give me a random unheard pop track".
- **Relevance gate:** off-topic messages ("hi", "who is the US president") get a polite canned refusal **without calling the model**.
- **Grounded answers:** the model only sees a small excerpt of relevant dataset rows and is told to answer only from them.
- **Streaming replies:** answers appear token by token via `st.write_stream`.
- **Scrollable chat window** with a fixed height, and a **Clear chat** button.

### App-wide
- **Bring your own data:** upload a Spotify-style CSV in the sidebar. No code changes needed.
- **Column auto-mapping** for common Kaggle variants (`artists` → `artist_name`, `track_genre` → `genre`, and so on).
- **Data cleaning and caching** (`@st.cache_data`) so the 114k-row file loads once.
- **Scope warning banner** so users know the limits of the AI.
- **No API key in the UI:** the token is read from Streamlit secrets.
- **Friendly error handling:** missing token, unreachable model, invalid JSON from the model and missing dataset columns all show a clear message instead of crashing.
- **Custom dark theme** with a single mood color system reused across the diagram, cards, badges and charts.

---

## 2. Model and tech stack

### Model
| | |
|---|---|
| **Model** | `Qwen/Qwen3-4B-Instruct-2507` |
| **Provider** | Hugging Face Inference API via `huggingface_hub.InferenceClient` |
| **Used for** | Both features: recommendation reasoning (JSON output) and the dataset chatbot (streamed text) |
| **Settings** | Recommendations: `max_tokens=1500`, `temperature=0.6`. Chatbot: `max_tokens=300`, `temperature=0.6`, `stream=True` |
| **Where to change it** | `DEFAULT_HF_MODEL` in `utils/ai_utils.py` |

### Stack
| Layer | Tool |
|---|---|
| UI | Streamlit |
| Data | Pandas |
| GenAI | Hugging Face Inference API (`huggingface_hub`) |
| Charts | Altair |
| Styling | Custom CSS injected through `utils/theme.py`, plus `.streamlit/config.toml` |
| Hosting | Streamlit Community Cloud |

---

## 3. Design decisions: the whys

**Why mood and popularity instead of a "time period" filter?**
The dataset has no release-year column, so a time-period filter was impossible. Valence (musical positivity) and energy (intensity) are in the data and map naturally onto a 2×2 of moods people already understand. Popularity gives a second axis between hits and hidden gems.

| valence | energy | mood |
|---------|--------|------|
| high (≥ 0.5) | high (≥ 0.5) | Happy & Energetic |
| high | low | Chill & Content |
| low | high | Angry & Intense |
| low | low | Sad & Mellow |

**Why filter with pandas before calling the model?**
114k tracks cannot fit in a prompt, and sending even thousands would be slow and costly. `filter_candidates()` scores every track and keeps the top 60, so the model receives a small, relevant list. This makes it fast and cheap and lets a small model do the job well.

**Why does the model only pick from the shortlist?**
Language models invent plausible-sounding songs. Restricting the model to a supplied list, and saying so in the prompt, keeps recommendations tied to real rows in the dataset.

**Why Hugging Face and one model for both features?**
Hugging Face offers free-tier inference and a large choice of open models, which suits a student project. Using one model for recommendations and the chatbot keeps the setup, token and code simple. `Qwen3-4B-Instruct-2507` is small enough for free-tier limits and follows "answer only from this context" instructions well. It can be swapped with one line.

**Why a keyword relevance gate for the chatbot?**
It is instant, free and predictable. Off-topic messages never reach the model, so no tokens are spent and the app cannot drift into general trivia. The gate checks music keywords, genre words and full artist names.

**Why match artists as full phrases, not single words?**
An early version added every word of every artist name to the vocabulary. With 30k+ artists, ordinary words like "weather" or "president" became "on-topic", so almost anything passed. Matching only full multi-word names restored a real signal.

**Why ground the chatbot in a 40-row dataset excerpt?**
Retrieval of a few relevant rows (matching genre, mood or artist words in the question, sorted by popularity) gives the model real data to cite, which reduces hallucination and keeps answers consistent with the dataset.

**Why stream the chatbot's answers?**
Streaming shows progress immediately instead of a long spinner followed by a wall of text, which feels much more responsive.

**Why a confirmation dialog when no favorites are picked?**
Without favorites the ranking falls back to popularity, which does not reflect anyone's taste. The dialog sets that expectation and lets the user back out.

**Why is the token in `secrets.toml` and not a text box?**
Typing keys into a UI is easy to leak and awkward for shared deployments. Secrets files are git-ignored and Streamlit Cloud has a matching Secrets panel, so users of the deployed app never see or need a key.

**Why show the candidate shortlist?**
It makes the pipeline transparent: users (and graders) can see what pandas selected and what the AI chose from.

**Why Altair charts and a dark theme with one mood color set?**
Altair gives clean, declarative charts that Streamlit renders natively. Reusing the same four mood colors for the diagram, card borders, badges and mood chart makes the color coding read as a system rather than decoration.

---

## 4. How it works

```
                 ┌────────────────────────────┐
 CSV (114k) ───▶ │ load_dataset(): clean,     │
                 │ rename, add mood, dedupe   │
                 └─────────────┬──────────────┘
                               │
        ┌──────────────────────┴───────────────────────┐
        ▼                                              ▼
 Tab 1: Recommendations                         Tab 2: Chatbot
 user favorites                                 user question
        │                                              │
 filter_candidates()                            is_dataset_related()
 score → top 60                                   │ no → canned refusal
        │                                         │ yes
 get_recommendations()                          build_chat_context()
 Hugging Face model → JSON                      top 40 relevant rows
        │                                              │
 cards + charts + shortlist                     Hugging Face model → streamed answer
```

### Recommendation scoring (`filter_candidates`)
| Signal | Points |
|---|---|
| Genre matches a chosen genre | +2 |
| Artist matches a chosen artist | +3 |
| Mood matches a chosen mood | +2 |
| Popularity within the chosen range | +1 |
| Popularity tiebreaker | up to +0.5 |

The top 60 by score become the shortlist. If nothing scores above zero, the full dataset is used as the pool.

### Data cleaning (`load_dataset`)
1. Read the CSV and drop a stray unnamed index column.
2. Rename common column variants.
3. Require `track_name`, `artist_name`, `genre` (clear error if missing) and drop rows missing them.
4. Strip whitespace from text columns.
5. Coerce numeric columns with `pd.to_numeric(errors="coerce")`.
6. Fill missing valence/energy with the median and compute `mood`.
7. Clip popularity to 0-100.
8. Drop duplicate (track, artist) pairs and reset the index.

---

## 5. Project structure

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
│   ├── data_utils.py           # load/clean, mood scoring, candidate filter, relevance gate
│   ├── ai_utils.py             # Hugging Face calls (recommendations + chatbot)
│   └── theme.py                # CSS, mood colors, badges, valence/energy diagram
└── .streamlit/
    ├── config.toml             # dark theme
    ├── secrets.toml            # your real HF_TOKEN (git-ignored)
    └── secrets.toml.example    # template with a placeholder token
```

### Recommended `requirements.txt`
```
streamlit>=1.38
pandas>=2.2
altair>=5
huggingface_hub>=0.24
```

---

## 6. Dataset

Bundled file: `data/spotify_dataset.csv` (114,000 rows × 21 columns). It includes track ID, artists, album, track name, popularity, duration, explicit flag, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time signature and genre.

Custom CSVs need at least:

| required | notes |
|---|---|
| `track_name` | song title |
| `artist_name` (or `artists`/`artist`) | artist |
| `genre` (or `track_genre`/`genres`) | genre label |

Optional numeric columns (`popularity`, `valence`, `energy`, `danceability`, `tempo`, and so on) are used when present. Without `valence` and `energy`, moods show as "Unknown".

---

## 7. Setup and run locally

**Requirements:** Python 3.10+ and a free Hugging Face account.

```bash
cd music_recommender_app
pip install -r requirements.txt
```

1. Create an access token at https://huggingface.co/settings/tokens.
2. Copy the template and add your token:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   ```toml
   # .streamlit/secrets.toml
   HF_TOKEN = "hf_..."
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

---

## 8. Using the app

**Get recommendations**
1. Open **🎯 Get Recommendations**.
2. Pick genres, artists and/or moods, then set the popularity range and number of tracks.
3. Click **Get recommendations**. If you picked nothing, confirm the dialog.
4. Read the cards (each has a reason), check the charts, and open the expander to see the shortlist.

**Chat with the dataset**
1. Open **💬 Ask the Dataset**.
2. Try "Recommend me a chill indie artist" or "Give me a random unheard pop track".
3. Use **Clear chat** to start over.

**Use your own data:** upload a CSV in the sidebar.

---

## 9. Deploy to Streamlit Community Cloud

1. Push the project folder to GitHub. `secrets.toml` is excluded by `.gitignore`.
2. Go to https://share.streamlit.io and click **New app**.
3. Select the repo, branch and `app.py`.
4. Open **Settings → Secrets** and paste:
   ```toml
   HF_TOKEN = "hf_..."
   ```
5. Deploy. Visitors never see or type a key.

---

## 10. Security notes

- Never commit `.streamlit/secrets.toml`. Keep only the placeholder `secrets.toml.example` in Git.
- Do not paste your real token into chats, screenshots or issue reports.
- If a token is ever exposed, revoke it at https://huggingface.co/settings/tokens and create a new one.
- Prefer a fine-grained, read-only token with inference permission only.

---

## 11. Testing and iteration

**Tested**
- Chatbot on-topic: "Recommend me a chill indie artist", "Give me a random unheard pop track".
- Chatbot off-topic: "hi", "who is the US president" (refused without a model call).
- Recommendations with only a genre, only an artist, only a mood, and all three together.
- No favorites selected (confirmation dialog).
- Narrow popularity ranges (0-20 and 80-100).
- Uploading a custom CSV.

**Improvements made**
- Artist names matched as full phrases (fixes false "on-topic" matches).
- Chat moved into a fixed-height scroll box.
- Streaming replaces the spinner-then-dump behavior.
- JSON fence stripping and a fallback when the model returns invalid JSON.
- Compact layout so the app fits a normal screen.

**Feedback loop:** share with classmates, note where recommendations or answers feel off, then tune scoring weights, prompts and the keyword list.

---

## 12. Known limitations

- **Dataset-only knowledge:** the app cannot recommend or discuss anything outside the loaded data.
- **No release year:** mood and popularity stand in for time period.
- **Mood is approximate:** it is a 0.5 cutoff on two audio features, so borderline tracks can land in either quadrant.
- **Prompt-level grounding:** the model is instructed to use only the supplied rows, but the app does not currently verify that every returned track is in the shortlist.
- **Small model:** a 4B model can occasionally return invalid JSON or a weaker reason. The app falls back gracefully but the output may be empty.
- **Keyword gate:** it can let through loosely music-related chat and can block on-topic questions that use no known keyword.
- **Limited context:** recommendations see 60 candidates and chat sees 40 rows, so results depend on the pandas step.
- **Free-tier rate limits:** the Hugging Face Inference API may throttle or be slow.

---

## 13. Next steps

- Validate that every recommended track exists in the shortlist.
- More filters: danceability, tempo, explicit-content toggle.
- "Similar to this track" recommendations using audio-feature distance.
- Cache model responses per unique query to reduce API calls.
- Replace the keyword gate with a small zero-shot classifier.
- Try other instruction-tuned models (for example `HuggingFaceTB/SmolLM2-1.7B-Instruct` for speed).

---

## 14. Activity 3 checklist

| Step | Where it is done |
|---|---|
| 1. Define app idea and dataset | Spotify Tracks dataset; recommender plus dataset chatbot |
| 2. Set up environment | Project structure in section 5 |
| 3. Load and clean with Pandas | `utils/data_utils.py` → `load_dataset()` |
| 4. Integrate GenAI | `utils/ai_utils.py` → Hugging Face `Qwen3-4B-Instruct-2507` |
| 5. Streamlit interface | `app.py` (multiselects, sliders, tabs, chat, dialog, uploader) |
| 6. Visualize results | Altair genre and mood charts, valence/energy diagram |
| 7. Test and iterate | Section 11 |
| 8. Deploy | Streamlit Community Cloud (section 9) |
| 9. Next goals | Category filters and dataset chatbot are implemented; further ideas in section 13 |
