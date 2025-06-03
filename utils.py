import os
import numpy as np
import librosa
import joblib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from streamlit_mic_recorder import mic_recorder
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import streamlit as st
import io

# Emotion mapping based on encoder categories
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}





def record_audio(save_as="temp_audio.wav"):
    """Record audio using mic_recorder, re-encode, and save clean WAV."""
    audio_data = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="🛑 Stop Recording",
        just_once=True,
        use_container_width=True
    )

    if audio_data and audio_data.get("bytes"):
        try:
            audio_bytes = audio_data["bytes"]
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_segment.export(save_as, format="wav")
            st.success("✅ Recording saved successfully.")
            return save_as
        except Exception as e:
            st.error(f"❌ Failed to convert audio: {e}")
            return None
    elif audio_data is not None:
        st.warning("⚠️ No audio data recorded.")
    return None





def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    return result

import soundfile as sf

def get_features_inference(path):
    import os
    import streamlit as st

    if not os.path.exists(path):
        st.error(f"❌ File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")

    try:
        # Use soundfile to load any audio, then slice it manually if needed
        data, sr = sf.read(path)
        if len(data.shape) > 1:  # stereo to mono
            data = data.mean(axis=1)

        # Trim or pad to 2.5 seconds (~sr*2.5 samples)
        max_samples = int(sr * 2.5)
        if len(data) < max_samples:
            padding = max_samples - len(data)
            data = np.pad(data, (0, padding), mode='constant')
        else:
            data = data[:max_samples]

        features = extract_features(data, sr)
        return np.array([features])
    except Exception as e:
        st.error(f"❌ Failed to process audio: {e}")
        raise RuntimeError(f"Audio decode error: {e}")

@st.cache_resource
def load_artifacts():
    try:
        model = load_model("emotion_model.h5", compile=False)
        scaler = joblib.load("standard_scaler.save")
        encoder = joblib.load("onehot_encoder.save")
        return model, scaler, encoder
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        raise
        
def predict_emotion(audio_path, model, scaler, encoder):
    raw_features = get_features_inference(audio_path)
    scaled_features = scaler.transform(raw_features)
    scaled_features = np.expand_dims(scaled_features, axis=2)
    prediction = model.predict(scaled_features)
    predicted_label = encoder.inverse_transform(prediction)
    emotion_probs = dict(zip(encoder.categories_[0], prediction[0]))
    return predicted_label[0][0], emotion_probs


import pandas as pd
from datetime import datetime
import os

def log_prediction(filename, emotion, log_file="emotion_log.csv"):
    """Append emotion prediction with timestamp to diary log."""
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Emotion": emotion
    }

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    df.to_csv(log_file, index=False)
