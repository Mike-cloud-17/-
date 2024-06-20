from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from joblib import load
from sklearn.preprocessing import StandardScaler
from googleapiclient.discovery import build
import random 
import re
import librosa

app = Flask(__name__)

# Загрузка модели
model = load('model/catboost_model.joblib')

scaler = load('model/scaler.joblib')

# Загрузка датасета
df_tracks = pd.read_csv('music_data.csv')

# Установка авторизации для Spotify
client_id = 'ec1186866257420dba1c18ec3198bd51'
client_secret = 'ae7b7c8276374decb0b9a2015673de38'
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Настройки YouTube API
youtube_api_key = 'AIzaSyAUuTaeOfmyd2SSK5NGYjUg116IcQDGFXU'
youtube = build('youtube', 'v3', developerKey=youtube_api_key)

# Genre mapping
genre_mapping = {
    1: 'kpop',
    2: 'indian',
    3: 'metal',
    4: 'hip_hop',
    5: 'classical',
    6: 'blues'
}

def get_track_features(input_value):
    # Проверка, является ли ввод URL-адресом Spotify
    match = re.match(r'https?://open.spotify.com/track/([a-zA-Z0-9]+)', input_value)
    if match:
        # Извлекаем ID трека из URL
        track_id = match.group(1)
        track_data = sp.track(track_id)
    else:
        # Ищем трек по названию, как раньше
        results = sp.search(q=input_value, type='track', limit=1)
        if not results['tracks']['items']:
            return None
        track_data = results['tracks']['items'][0]

    track_id = track_data['id']
    track_features = sp.audio_features(track_id)[0]

    if track_features is None:
        return None

    features = {
        'duration_ms': track_data['duration_ms'],
        'explicit': track_data['explicit'],
        'danceability': track_features['danceability'],
        'energy': track_features['energy'],
        'key': track_features['key'],
        'loudness': track_features['loudness'],
        'mode': track_features['mode'],
        'speechiness': track_features['speechiness'],
        'acousticness': track_features['acousticness'],
        'instrumentalness': track_features['instrumentalness'],
        'liveness': track_features['liveness'],
        'valence': track_features['valence'],
        'tempo': track_features['tempo'],
        'time_signature': track_features['time_signature']
    }
    return features

def predict_genre(scaled_features):
    # Модель ожидает получить массив, поэтому нам не нужно его преобразовывать
    predicted = model.predict(scaled_features)

    # Возьмем первый элемент массива, если массивов много
    genre_num = predicted[0]

    if isinstance(genre_num, np.ndarray):
        # Если genre_num все еще массив, возьмем первый элемент из него
        genre_num = genre_num[0]

    # Теперь genre_num должно быть числом
    genre = genre_mapping.get(genre_num, "Unknown")
    return genre

def find_closest_track(features, genre, df_tracks):
    genre_column_name = 'genre_track'

    genre_tracks = df_tracks[df_tracks[genre_column_name] == genre]
    if genre_tracks.empty:
        print(f"No tracks found for genre: {genre}")
        return None

    feature_cols = [
        'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness',
        'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'time_signature'
    ]

    print(f"Found {len(genre_tracks)} tracks for genre: {genre}")

    # Calculate distances
    genre_tracks['distance'] = genre_tracks.apply(
        lambda row: np.linalg.norm(
            np.array([row[col] for col in feature_cols]) - 
            np.array([features[col] for col in feature_cols])
        ) if not any(np.isnan([row[col] for col in feature_cols])) else np.nan,
        axis=1
    )

    # Check for NaN values in distance column
    if genre_tracks['distance'].isna().all():
        print(f"All distances are NaN for genre: {genre}")
        return None

    # Sort tracks by distance and get top 10 closest
    closest_tracks = genre_tracks.nsmallest(10, 'distance')

    # Select one random track from the top 10
    if not closest_tracks.empty:
        selected_track = closest_tracks.sample(n=1).iloc[0]
        print(f"Selected closest track ID: {selected_track['track_id']} with distance {selected_track['distance']}")
        return selected_track
    else:
        return None

def get_track_details(track_id):
    track_data = sp.track(track_id)
    track_name = track_data['name']
    artists = ', '.join([artist['name'] for artist in track_data['artists']])
    return track_name, artists

def get_youtube_link(track_name):
    search_response = youtube.search().list(
        q=track_name,
        part='snippet',
        maxResults=1
    ).execute()

    # Проверяем, есть ли элементы в ответе и есть ли videoId
    if search_response.get('items') and 'videoId' in search_response['items'][0]['id']:
        video_id = search_response['items'][0]['id']['videoId']
        youtube_iframe = f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
        return youtube_iframe
    else:
        # Если videoId не найден, предоставляем ссылку на поиск YouTube
        youtube_search_url = f'https://www.youtube.com/results?search_query={track_name}'
        return f'<a href="{youtube_search_url}" target="_blank">Watch on YouTube</a>'  # Возвращает гиперссылку на поиск YouTube

def get_spotify_link(track_id):
    return f'<iframe src="https://open.spotify.com/embed/track/{track_id}" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'

@app.route('/next', methods=['POST'])
def next_track():
    genre = request.form['genre']
    current_track_id = request.form['track_id']
    genre_tracks = df_tracks[df_tracks['genre_track'] == genre]
    genre_tracks = genre_tracks[genre_tracks['track_id'] != current_track_id]

    if not genre_tracks.empty:
        next_track = genre_tracks.sample(n=1).iloc[0]
        track_id = next_track['track_id']
        track_name, artists = get_track_details(track_id)
        youtube_link = get_youtube_link(track_name)
        spotify_link = get_spotify_link(track_id)  # Получение Spotify ссылки
        result = {
            'genre': genre,
            'track_name': track_name,
            'artists': artists,
            'youtube_link': youtube_link,
            'spotify_link': spotify_link  # Добавление в результат
        }
        return jsonify(result)
    else:
        return jsonify({'error': 'No other tracks available in this genre'})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        track_name = request.form['track_name']
        features = get_track_features(track_name)
        if features:
            # Преобразуем и масштабируем признаки, получаем предсказание жанра
            features_list = [features[feature] for feature in [
                'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness',
                'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                'valence', 'tempo', 'time_signature'
            ]]
            features_array = np.array([features_list], dtype=float)
            scaled_features = scaler.transform(features_array)
            genre = predict_genre(scaled_features)
            closest_track = find_closest_track(features, genre, df_tracks)
            
            if closest_track is not None and not closest_track.empty:
                track_id = closest_track['track_id']
                track_name, artists = get_track_details(track_id)
                youtube_link = get_youtube_link(track_name)
                spotify_link = get_spotify_link(track_id)
                return render_template('index.html', track_name=track_name, artists=artists, genre=genre, youtube_link=youtube_link, spotify_link=spotify_link, track_id=track_id, form_visible=True)
            else:
                return render_template('index.html', error="No similar tracks found.", form_visible=True)
        else:
            return render_template('index.html', error="Track not found on Spotify.", form_visible=True)
    return render_template('index.html', form_visible=True)


if __name__ == '__main__':
    app.run(debug=True)

