# 🎧 Spotify Mood Playlist Generator

A full-stack project that analyzes user input text, detects mood using NLP, and generates a Spotify playlist based on the detected mood.

## 🚀 Features
- Mood detection using a Hugging Face transformer model  
- Playlist generation from a labeled Spotify dataset  
- FastAPI backend with REST API endpoints  
- Fallback logic if NLP model is unavailable  
- CORS enabled for frontend integration  
- Supports moods: happy, sad, energetic, calm, neutral  

## 🧠 How It Works
User submits a text input → backend analyzes the text using a transformer model → detected emotion is mapped to a simplified mood category → songs matching that mood are sampled from the dataset → Spotify track links are returned as a playlist.

## 📁 Project Structure
spotify-playlist-generator/  
├── backend/  
│   ├── app.py  
│   ├── requirements.txt  
│   ├── data/ (not included if dataset is large)  
├── frontend/  
│   ├── index.html  
│   ├── package.json  
│   ├── tailwind.config.js  
├── .gitignore  
├── README.md  

## ⚙️ Setup Instructions

Clone the repository:
git clone https://github.com/YOUR_USERNAME/spotify-playlist-generator.git  
cd spotify-playlist-generator/backend  

Create a virtual environment:
python -m venv venv  
source venv/bin/activate   # Mac/Linux  
venv\Scripts\activate      # Windows  

Install dependencies:
pip install -r requirements.txt  

Download the dataset (not included due to size) from your provided link and place it in:
backend/data/278k_labelled_uri.csv  

Run the backend server:
uvicorn app:app --reload  

Server will run at:
http://127.0.0.1:8000  

## 📡 API Endpoints

GET /  
Returns: { "message": "Mood Playlist API is running" }

POST /analyze_mood  
Request:
{ "text": "I feel really happy today" }

Response:
{
  "detected_mood": "happy",
  "raw_label": "joy",
  "confidence": 0.98,
  "playlist": [
    {
      "track_uri": "spotify:track:...",
      "spotify_url": "https://open.spotify.com/track/..."
    }
  ]
}

## 📦 Dependencies
FastAPI, Uvicorn, Pandas, Transformers, PyTorch, Spotipy, Python-dotenv

Install with:
pip install -r requirements.txt

## ⚠️ Notes
- Dataset is excluded due to size limitations  
- If transformers/torch is not installed, the app falls back to random mood selection  
- CORS is open for development purposes  
- Restrict origins in production  

## 🛠 Future Improvements
- Spotify OAuth integration  
- Frontend UI (React or Streamlit)  
- Deployment to cloud (Render/AWS)  
- Improved emotion classification accuracy  
- Model caching for faster startup  
- User-specific playlists  
