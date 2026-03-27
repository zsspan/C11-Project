import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET"
))

df = pd.read_csv("../../datasets/track_data_final.csv")
track_ids = df["track_id"].tolist()

# fetch in batches of 100
features = []
for i in range(0, len(track_ids), 100):
    batch = track_ids[i:i+100]
    results = sp.audio_features(batch)
    
    # filter out 'None' values returned by Spotify for tracks with missing data
    valid_results = [track for track in results if track is not None]
    features.extend(valid_results)

features_df = pd.DataFrame(features)

# merge back on track_id
df = df.merge(features_df[["id", "danceability", "energy", "acousticness",
                             "instrumentalness", "valence", "tempo", "speechiness"]],
              left_on="track_id", right_on="id", how="left").drop(columns="id")

df.to_csv("../datasets/full_spotify_dataset.csv", index=False)