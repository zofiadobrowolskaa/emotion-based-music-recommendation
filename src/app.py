import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import ast
import random
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# load environment variables for spotify api
load_dotenv()

st.set_page_config(page_title="Emotion Music Recommender", layout="wide")

EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# load this resource only once and reuse it across app reruns
@st.cache_resource
def load_ai_models():
    # load mtcnn for face detection (cached to save ram and time)
    detector = MTCNN()
    # load final trained mobilenetv2 model
    emotion_model = load_model("../models/final_emotion_model.h5")
    return detector, emotion_model

@st.cache_data
def load_association_rules():
    rules_path = "../results/music_association_rules.csv"
    if os.path.exists(rules_path):
        return pd.read_csv(rules_path)
    return None

# load models and rules into memory
detector, emotion_model = load_ai_models()
rules_df = load_association_rules()

# initialize spotify client
def init_spotify():
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        return None
        
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=auth_manager)

sp = init_spotify()

def process_and_predict(image_pil):
    # convert pil image to numpy array in rgb format
    img_array = np.array(image_pil)
    
    # detect faces using mtcnn
    faces = detector.detect_faces(img_array)
    
    if not faces:
        return None, None, "No face detected in the image. Please try another one."
        
    # assume the first detected face is the primary one
    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)
    
    # crop the face and resize to match model input (48x48)
    cropped_face = img_array[y:y+h, x:x+w]
    resized_face = cv2.resize(cropped_face, (48, 48))
    
    # normalize exactly like in training phase
    normalized_face = resized_face.astype('float32') / 255.0
    input_tensor = np.expand_dims(normalized_face, axis=0)
    
    # 1. get raw predictions from the model
    raw_predictions = emotion_model.predict(input_tensor)[0]
    
    # 2. apply heuristic prior adjustment to fix dataset bias and mtcnn cropping differences
    calibration_weights = np.array([
        1.2,  # Angry
        4.0,  # Disgust
        3.5,  # Fear
        0.7,  # Happy
        2.0,  # Neutral
        1.2,  # Sad
        3.0   # Surprise
    ])
    
    # apply weights
    calibrated_predictions = raw_predictions * calibration_weights
    
    # 3. re-normalize so probabilities sum back to 1.0 (100%)
    final_predictions = calibrated_predictions / np.sum(calibrated_predictions)
    
    return cropped_face, final_predictions, None

def parse_consequents(val_string):
    # safely convert string representation of lists from csv into python list
    if pd.isna(val_string):
        return []
    if val_string.startswith('['):
        try:
            return ast.literal_eval(val_string)
        except:
            return []
    return [val_string]

def get_spotify_recommendations(emotion):
    # fetch music based on emotion and apriori rules
    if sp is None:
        return None, "Spotify API keys missing. Check your .env file."
        
    if rules_df is None:
        return None, "Apriori rules file not found. Run association_rules.py first."
        
    # get the best rule for the detected emotion
    emotion_rules = rules_df[rules_df['antecedents'] == emotion.lower()]
    
    if emotion_rules.empty:
        return None, f"No association rules found for emotion: {emotion}"
        
    # get attributes from the highest confidence rule
    top_rule = emotion_rules.iloc[0]['consequents']
    attributes = parse_consequents(top_rule)
    
    # map specific emotions to base spotify genres
    genre_mapping = {
        'happy': 'pop', 'sad': 'acoustic', 'angry': 'metal', 
        'fear': 'ambient', 'surprise': 'edm', 'disgust': 'grunge', 'neutral': 'chill'
    }
    seed_genre = genre_mapping.get(emotion.lower(), 'pop')
    
    # map Apriori rules to semantic search queries
    query_terms = [seed_genre, emotion.lower()]
    
    if 'high_energy' in attributes or 'fast_tempo' in attributes:
        query_terms.append("upbeat")
    if 'low_energy' in attributes or 'slow_tempo' in attributes:
        query_terms.append("relax")
    if 'minor_key' in attributes:
        query_terms.append("dark")
        
    search_query = " ".join(query_terms)
    
    try:
        # fetch tracks from spotify api and process recommendation logic in python
        results = sp.search(q=search_query, type='track')
        tracks = results['tracks']['items']
        
        if tracks:
            # pick 3 random tracks to ensure fresh recommendations every time
            sample_size = min(3, len(tracks))
            selected_tracks = random.sample(tracks, sample_size)
            return selected_tracks, None
        else:
            return None, "No matching tracks found on Spotify."
    except Exception as e:
        return None, f"Spotify API error: {str(e)}"


st.title("Emotion-Based Music Recommendation 🎵")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Image 📸")
    input_mode = st.radio("Choose input source:", ["Upload Image", "Live Camera"])
    
    image = None
    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Select a photo from your device...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
    else:
        camera_photo = st.camera_input("Take a picture of your face")
        if camera_photo:
            image = Image.open(camera_photo).convert('RGB')

    if image:
        st.image(image, caption="Selected Image for Analysis", width='stretch')

with col2:
    st.header("Analysis & Recommendations 📊")
    
    if image:
        st.subheader("Detected Emotion")
        
        with st.spinner('Analyzing face geometry and emotions...'):
            cropped_face, probabilities, error_msg = process_and_predict(image)
            
        if error_msg:
            st.error(error_msg)
        else:
            st.image(cropped_face, caption="Extracted Face", width=150)
            
            # find the index of the highest probability
            dominant_index = np.argmax(probabilities)
            dominant_emotion = EMOTION_CLASSES[dominant_index]
            
            st.success(f"Dominant Emotion: **{dominant_emotion}**")
            
            # generate dynamic bar chart using pandas dataframe
            st.markdown("**Probability Distribution:**")
            prob_df = pd.DataFrame({
                "Emotion": EMOTION_CLASSES,
                "Probability (%)": probabilities * 100
            }).set_index("Emotion")
            st.bar_chart(prob_df)
            
            st.markdown("---")
            st.subheader("Music Recommendations 🎧")

            with st.spinner("Consulting Apriori rules and fetching from Spotify..."):
                tracks, spotify_err = get_spotify_recommendations(dominant_emotion)
                
            if spotify_err:
                st.error(spotify_err)

            elif tracks:
                track_cols = st.columns(3)

                 # iterate through recommended tracks together with their index
                for idx, track in enumerate(tracks):

                    # place each track inside its corresponding column
                    with track_cols[idx]:
                        # extract metadata
                        cover_url = track['album']['images'][0]['url'] if track['album']['images'] else "https://via.placeholder.com/150"
                        track_name = track['name']
                        artist_name = track['artists'][0]['name']
                        spotify_url = track['external_urls']['spotify']
                        
                        # render custom card
                        st.image(cover_url, width='stretch')
                        st.markdown(f"**{track_name}**")
                        st.caption(f"By: {artist_name}")
                        st.markdown(f"[Listen on Spotify]({spotify_url})", unsafe_allow_html=True)
            else:
                st.info("No matching tracks found for these specific rules.")
    else:
        st.warning("Please provide an image using the controls on the left to start.")