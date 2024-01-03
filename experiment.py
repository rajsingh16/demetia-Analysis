'''#import pyAudioAnalysis
from pyAudioAnalysis import audioSegmentation
import hmmlearn.hmm

def get_pause_masks(file):
    frame_duration_ms = 30

    audio, sample_rate = read_wave(file)
    #vad = webrtcvad.Vad()

    vad = audioSegmentation.vad_collector(sample_rate, frame_duration_ms)

    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, frame_duration_ms, 10, vad, frames)

    segments = np.asarray(segments)
    # segments = (segments - np.mean(segments))/np.std(segments)
    # print(segments)
    return segments'''
'''import librosa
import numpy as np

def get_pause_masks(file, frame_duration_ms=30, energy_threshold=0.1):
    audio, sample_rate = librosa.load(file, sr=None, mono=True)  # Load audio using LibROSA

    frame_length = int(frame_duration_ms / 1000 * sample_rate)
    hop_length = int(frame_length / 2)

    # Compute root mean square (RMS) energy of audio frames
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Apply energy threshold to identify speech segments
    speech_segments = np.where(rms > energy_threshold)[0]

    # Convert speech segments into time intervals
    segment_intervals = librosa.frames_to_time(speech_segments, sr=sample_rate, hop_length=hop_length)

    return segment_intervals'''

'''import os

folder_path = 'D:\\projects\\dmentia project\\ADReSS-IS2020-train\\ADReSS-IS2020-data\\train\\transcription\\cc'  # Replace with the actual path to the folder

# List all files and folders in the specified folder
files = os.listdir(folder_path)

# Iterate over the files and folders
for file in files:
    # Perform desired operations on each file or folder
    print(file)'''
import os
import pickle
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Specify the directory where the audio files are located
audio_directory = 'D:/projects/dmentia project/ADReSS-IS2020-train/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd'

# Specify the directory where you want to save the scaler and PCA files
output_directory = 'C:/Users/Admin/IdeaProjects/dementia analysis/.idea/models/testing_silence/3_c/compare'

# Initialize a counter
counter = 1

# Iterate through the audio files in the directory
for filename in os.listdir(audio_directory):
    if filename.endswith('.wav'):
        # Load and preprocess audio data
        audio_path = os.path.join(audio_directory, filename)
        audio_data, sr = librosa.load(audio_path, sr=None)
        # Preprocess audio data as needed

        # Apply feature extraction
        features = librosa.feature.mfcc(y=audio_data, sr=sr)
        # Apply other feature extraction techniques as desired

        # Scale the features using a scaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Perform dimensionality reduction using PCA
        pca = PCA(n_components=3)  # Set the desired number of components
        pca_features = pca.fit_transform(scaled_features)

        # Save the scaler and PCA objects with sequential file names
        scaler_filename = os.path.join(output_directory, f'scaler{counter}.pkl')
        pca_filename = os.path.join(output_directory, f'pca{counter}.pkl')
        pickle.dump(scaler, open(scaler_filename, 'wb'))
        pickle.dump(pca, open(pca_filename, 'wb'))

        # Increment the counter
        counter += 1
