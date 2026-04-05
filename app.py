import sys
import os
import random

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    from fastapi.middleware.cors import CORSMiddleware
    import pandas as pd
    # Try to import transformers if available; it's optional — we'll fall back if missing
    try:
        from transformers import pipeline
    except Exception:
        pipeline = None
except ModuleNotFoundError as e:
    missing = e.name
    msg = (
        f"Missing Python dependency: {missing}.\n"
        "Please install required packages into your virtualenv:\n"
        "  python -m pip install -r requirements.txt\n"
        "Or activate your venv and run the same command.\n"
        "If you don't have a venv, create one with:\n"
        "  python -m venv venv && source venv/bin/activate\n"
    )
    print(msg, file=sys.stderr)
    sys.exit(1)

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SONG_URI_CSV = os.path.join(DATA_DIR, "278k_labelled_uri.csv")

# -------------------------------
# Load dataset
# -------------------------------
if not os.path.exists(SONG_URI_CSV):
    raise FileNotFoundError(f"CSV not found at {SONG_URI_CSV}")

uri_df = pd.read_csv(SONG_URI_CSV)

# Detect Spotify URI column
possible_uri_columns = ["track_uri", "uri", "spotify_uri", "id"]
uri_column = next((col for col in possible_uri_columns if col in uri_df.columns), None)

if uri_column is None:
    raise ValueError(f"No Spotify URI column found. Columns: {uri_df.columns.tolist()}")

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Mood Playlist API")

# Allow React frontend to access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Request model
# -------------------------------
class MoodRequest(BaseModel):
    text: str

# -------------------------------
# Mood mapping
# -------------------------------
emotions = ["sad", "happy", "energetic", "calm", "neutral"]
mood_map = {"sad": 0, "happy": 1, "energetic": 2, "calm": 3, "neutral": 4}

# Optional HF classifier (initialized below if transformers available)
classifier = None

# Mapping from HF emotion labels to our mood categories
HF_TO_MOOD = {
    "sadness": "sad",
    "joy": "happy",
    "anger": "energetic",
    "surprise": "energetic",
    "fear": "calm",
    "love": "happy",
}


def map_hf_label_to_mood(label: str) -> str | None:
    """Map a Hugging Face emotion label to one of our simple moods.

    Uses exact mapping first, then simple heuristics for unseen labels.
    Returns None if no reasonable mapping is found.
    """
    if not label:
        return None
    key = label.lower()
    if key in HF_TO_MOOD:
        return HF_TO_MOOD[key]

    # Heuristic substring checks
    if "joy" in key or "happy" in key or "love" in key or "elation" in key:
        return "happy"
    if "sad" in key or "depress" in key or "down" in key:
        return "sad"
    if "ang" in key or "rage" in key or "annoy" in key or "frustr" in key:
        return "energetic"
    if "fear" in key or "anx" in key or "scare" in key:
        return "calm"
    if "surpris" in key or "excite" in key or "enth" in key:
        return "energetic"
    if "neutral" in key or "calm" in key or "bore" in key:
        return "neutral"

    return None

# -------------------------------
# Test root endpoint
# -------------------------------
@app.get("/")
def root():
    return {"message": "Mood Playlist API is running"}

# -------------------------------
# Analyze mood endpoint
# -------------------------------
@app.post("/analyze_mood")
def analyze_mood(request: MoodRequest):
    user_text = request.text.strip()
    if not user_text:
        return {"error": "No text provided"}
    # Prefer a transformer-based classifier if available
    dominant_emotion = None
    if 'pipeline' in globals() and pipeline is not None:
        try:
            # initialize pipeline lazily to avoid heavy startup cost if module is present but not used
            global classifier
            if classifier is None:
                # model choice: a small emotion classifier; may be changed to any other HF model
                classifier = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion", return_all_scores=False)

            result = classifier(user_text[:512])
            # result can be a list or dict depending on transformers version
            if isinstance(result, list) and len(result) > 0:
                res = result[0]
            else:
                res = result

            label_name = res.get("label") if isinstance(res, dict) else None
            score = res.get("score") if isinstance(res, dict) else None
            if label_name:
                dominant_emotion = label_name.lower()
            confidence = float(score) if score is not None else None
        except Exception:
            dominant_emotion = None

    # Fallback to random if transformer not available or failed
    if not dominant_emotion:
        dominant_emotion = random.choice(emotions)
        confidence = None

    # Map HF labels to our simple mood categories where possible
    # Try to map the HF label to our mood categories using helper
    mapped_mood = map_hf_label_to_mood(dominant_emotion) if dominant_emotion else None
    label = mood_map.get(mapped_mood, 4)

    # Choose a detected_mood to return that the frontend can use for theming.
    detected_mood_for_response = mapped_mood or (dominant_emotion if dominant_emotion in mood_map else None)


    # Filter songs by labels (lowercase, as in CSV)
    if "labels" in uri_df.columns:
        mood_songs = uri_df[uri_df["labels"] == label]
    else:
        mood_songs = uri_df

    if len(mood_songs) == 0:
        mood_songs = uri_df

    # Pick up to 10 random songs
    recommendations = mood_songs.sample(min(10, len(mood_songs)))

    # Build playlist with Spotify URLs
    playlist = [
        {"track_uri": uri, "spotify_url": f"https://open.spotify.com/track/{str(uri).split(':')[-1]}"}
        for uri in recommendations[uri_column]
    ]

    return {
        "detected_mood": detected_mood_for_response,
        "raw_label": dominant_emotion,
        "confidence": confidence,
        "playlist": playlist
    }


if __name__ == "__main__":
    try:
        import uvicorn
    except ModuleNotFoundError:
        print("uvicorn is not installed. Install with: python -m pip install uvicorn[standard]", file=sys.stderr)
        sys.exit(1)

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
