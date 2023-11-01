import streamlit as st
import soundfile as sf
import librosa
import numpy as np
import pickle

# Load the trained model
model_filename = 'modelForPrediction1.sav'
loaded_model = pickle.load(open(model_filename, 'rb'))

# Define emotions
emotions = {
    'calm': "😌",
    'happy': "😁",
    'fearful': "😨",
    'disgust': "😒",
}

# Function to predict emotions
def predict_emotion(audio_file):
    feature = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    feature = feature.reshape(1, -1)
    prediction = loaded_model.predict(feature)
    return prediction[0]

# Function to extract features from an audio file
def extract_feature(file_name, mfcc, chroma, mel):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# Streamlit app
st.title("Speech Emotion Recognition Web APP")

# Upload an audio file
audio_file = st.file_uploader("Upload an audio file (.WAV)", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')

    if st.button("Predict Emotion"):
        prediction = predict_emotion(audio_file)
        emotion_icon = emotions.get(prediction, "Unknown")
        st.success(f"Predicted Emotion: ㅤㅤㅤㅤㅤㅤㅤㅤㅤ {prediction} {emotion_icon}")
        st.write("Accuracy of Current Model is 76.04%")
        st.balloons()

st.sidebar.markdown("## About this Project")
st.sidebar.write(""" #### Project Overview 📊

Welcome to our Speech Emotion Recognition project! 🎙️ Our system utilizes the power of Machine Learning and feature extraction techniques to identify human emotions based on voice recordings. We've employed the MLP (Multi-Layer Perceptron) classifier, a robust algorithm trained using Backpropagation.

🧬 Feature Extraction : -
                 
Our system extracts crucial audio features such as Mel-Frequency Cepstral Coefficients (MFCCs), chroma, and mel-spectrogram to capture the essence of speech patterns.

🎯 Our mission is to enhance the understanding of human emotions through sound, contributing to a wide range of applications, from voice assistants to mental health support.

Explore and experience the world of Speech Emotion Recognition with us! 🌟
""")

st.sidebar.info("Created by Group 2")
st.sidebar.write("Anuj - Maithili - Deepali - Lekha - Adarsh ")
