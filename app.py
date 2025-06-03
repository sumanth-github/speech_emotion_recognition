import streamlit as st
import json
import pandas as pd
import plotly.express as px
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils import load_artifacts, predict_emotion,record_audio,log_prediction
import os
# ---------- 1. Page Config ----------
st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")
st.title("🎙️ Speech Emotion Recognition")

# ---------- 2. Project Info ----------
with st.expander("ℹ️ About This Project", expanded=False):
    st.markdown("""
    **Speech Emotion Recognition (SER)** is an application of machine learning and signal processing that identifies human emotions from voice data.
    
    This project:
    - Uses a trained neural network model to classify emotions as `happy`, `sad`, `angry`, or `neutral`
    - Visualizes emotion confidence using a **radar chart**
    - Recommends AI-driven mood-based content
    - Built with `Python`, `Streamlit`, `Librosa`, and `Keras`
    """)

# ---------- 3. Load Artifacts ----------
model, scaler, encoder = load_artifacts()

# ---------- 4. UI Tabs ----------
tab1, tab2, tab3,tab4 = st.tabs(["🎤 Upload & Analyze", "📊 Audio Graph", "🤖 Suggestions","📔 Diary"])

# ========== TAB 1 ==========

with tab1:
    st.header("🎤 Record Audio or Upload File")

    col1, col2 = st.columns(2)
    audio_path = None

    with col1:
        st.markdown("#### 🎙️ Record via Microphone")
        st.info("Press 'Start' and 'Stop' to record.")
        mic_audio_path = record_audio("temp_audio.wav")  # Always attempt mic recording
        if mic_audio_path:
            audio_path = mic_audio_path  # Prioritize mic if recorded

    with col2:
        st.markdown("#### 📁 Upload an Audio File")
        uploaded_file = st.file_uploader("Choose a `.wav` or `.mp3` file", type=["wav", "mp3"])
        if uploaded_file:
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            audio_path = "temp_audio.wav"

    # Now handle audio prediction once, regardless of source
    if audio_path and os.path.exists(audio_path):
        st.audio(audio_path)

        with st.spinner("Analyzing emotion..."):
            emotion, emotion_probs = predict_emotion(audio_path, model, scaler, encoder)

        st.success(f"**Predicted Emotion:** `{emotion}`")
        log_prediction(audio_path, emotion)

        st.subheader("🎯 Emotion Confidence Radar")
        labels = list(emotion_probs.keys()) + [list(emotion_probs.keys())[0]]
        values = list(emotion_probs.values()) + [list(emotion_probs.values())[0]]

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            line=dict(width=2, color='royalblue')
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            width=500,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)


# ========== TAB 2 ==========
with tab2:
    st.header("Waveform Graph")

    try:
        data, sr = librosa.load("temp_audio.wav", duration=2.5, offset=0.6)

        fig1, ax1 = plt.subplots(figsize=(4,2))
        librosa.display.waveshow(data, sr=sr, ax=ax1)
        ax1.set_title("Waveform")
        st.pyplot(fig1)

    except:
        st.warning("⚠️ Please upload audio in the first tab to see graphs.")

# ========== TAB 3 ==========
with tab3:
    st.header("AI-Based Suggestions")

    try:
        st.info(f"Detected Emotion: **{emotion}**")

        if emotion == "sad":
            st.write("- 🌞 [Top Motivational Speech](https://www.youtube.com/watch?v=mgmVOuLgFB0)")
            st.write("- 🎧 [Relaxing Piano Music](https://www.youtube.com/watch?v=Mk7tu5QHf0Y)")
            st.write("- 📓 Try journaling your thoughts.")
        elif emotion == "angry":
            st.write("- 🧘 [5-Minute Guided Meditation](https://www.youtube.com/watch?v=inpok4MKVLM)")
            st.write("- 🔥 [How to Control Anger](https://www.youtube.com/watch?v=I3dFeFzY5gM)")
            st.write("- 🚶 Take a walk and breathe deeply.")
        elif emotion == "happy":
            st.write("- 🎉 [Uplifting Moments Compilation](https://www.youtube.com/watch?v=3PZ65s5QbUo)")
            st.write("- 🎵 [Feel-Good Music](https://www.youtube.com/watch?v=fHI8X4OXluQ)")
            st.write("- 💬 Share your happiness with someone!")
        elif emotion == "neutral":
            st.write("- ☕ [Lo-Fi Chill Beats](https://www.youtube.com/watch?v=2OEL4P1Rz04)")
            st.write("- 💡 Take a short break to stretch.")
            st.write("- 📚 Read a quick article.")
        else:
            st.write("Emotion not clearly recognized. Try another sample.")
    except:
        st.warning("⚠️ Please analyze audio in Tab 1 before using suggestions.")



with tab4:
    st.header("📔 Emotion Diary")

    log_file = "emotion_log.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)

        # Show latest logs
        st.markdown("### 🕒 Recent Predictions")
        st.dataframe(df.tail(10))

        # Pie chart
        st.markdown("### 📊 Emotion Distribution")
        fig = px.pie(df, names="Emotion", title="Overall Emotion Frequency")
        st.plotly_chart(fig)

        # Download
        st.markdown("### ⬇️ Download Diary Log")
        st.download_button(
            label="Download as CSV",
            data=df.to_csv(index=False),
            file_name="emotion_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No logs yet. Start analyzing some audio!")


# ---------- 5. Optional Sidebar ----------

with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("This app was built for showcasing ML-based emotional intelligence from voice.")
    st.markdown("Check out more projects at: [GitHub](https://github.com/yourrepo)")
