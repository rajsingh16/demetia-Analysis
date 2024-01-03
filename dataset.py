'''import glob
import os
import math
import time
import re
import csv
from config import config
import numpy as np
import sys
import types

np.random.seed(0)
p = np.random.permutation(108)  # n_samples = 108
p_subjects = np.random.RandomState(seed=0).permutation(242)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import collections
import contextlib
import sys
import wave
import audioop
import utils
import librosa

dataset_dir=r'D:\projects\dmentia project\ADReSS-IS2020-train\ADReSS-IS2020-data\train'

def get_pause_features(transcription_filename, audio_filename, audio_length_normalization=10):

    audio_len = utils.get_audio_length(audio_filename) / audio_length_normalization

    with open(transcription_filename, 'r',encoding = 'utf-8') as f:
        content = f.read()
        word_rate = utils.words_count(content) / (50 * audio_len)
        pause_rates = utils.get_pauses_cnt(content) / audio_len
        inv_rate = utils.get_n_interventions(content) / audio_len

    pause_features = np.concatenate(([inv_rate], pause_rates, [word_rate]), axis=-1)

    return pause_features
def get_intervention_features(transcription_filename, max_length=40):

    speaker_dict = {
        'INV': [0, 0, 1],
        'PAR': [0, 1, 0],
        'padding': [1, 0, 0]
    }

    with open(transcription_filename, 'r',encoding='utf-8') as f:
        content = f.read()
        content = content.split('\n')
        speakers = []

        for c in content:
            if 'INV' in c:
                speakers.append('INV')
            if 'PAR' in c:
                speakers.append('PAR')

        PAR_first_index = speakers.index('PAR')
        PAR_last_index = len(speakers) - speakers[::-1].index('PAR') - 1
        intervention_features = speakers[PAR_first_index:PAR_last_index]

    intervention_features = list(map(lambda x: speaker_dict[x], intervention_features))

    if len(intervention_features) > max_length:
        intervention_features = intervention_features[:max_length]
    else:
        pad_length = max_length - len(intervention_features)
        intervention_features = intervention_features + [speaker_dict['padding']] * pad_length

    return intervention_features
def get_spectogram_features(spectogram_filename):

    mel = np.load(spectogram_filename)
    # mel = feature_normalize(mel)
    mel = np.expand_dims(mel, axis=-1)
    return mel
csv.field_size_limit(2147483647)
def get_compare_features(compare_filename):
    compare_features = []
    with open(compare_filename, 'r') as file:
        content = csv.reader(file)
        for row in content:
            compare_features = row
    compare_features_floats = [float(item) for item in compare_features[1:-1]]
    return compare_features_floats

def prepare_data(dataset_dir, config):
    cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))
    all_speakers_cc = []
    for filename in cc_files:
        all_speakers_cc.append(get_intervention_features(filename, config.longest_speaker_length))
    cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))
    all_speakers_cd = []
    for filename in cd_files:
        all_speakers_cd.append(get_intervention_features(filename, config.longest_speaker_length))
    y_cc = np.zeros((len(all_speakers_cc), 2))
    y_cc[:, 0] = 1
    y_cd = np.zeros((len(all_speakers_cd), 2))
    y_cd[:, 1] = 1
    X_intervention = np.concatenate((all_speakers_cc, all_speakers_cd), axis=0).astype(np.float32)
    y_intervention = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
    filenames_intervention = np.concatenate((cc_files, cd_files), axis=0)
    y_reg_cc = utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
    y_reg_cd = utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))
    y_reg_intervention = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
    X_reg_intervention = np.copy(X_intervention)
    cc_transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))
    cc_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.wav')))
    all_counts_cc = []
    for t_f, a_f in zip(cc_transcription_files, cc_audio_files):
        pause_features = get_pause_features(t_f, a_f)
        all_counts_cc.append(pause_features)
    cd_transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))
    cd_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.wav')))
    all_counts_cd = []
    for t_f, a_f in zip(cd_transcription_files, cd_audio_files):
        pause_features = get_pause_features(t_f, a_f)
        all_counts_cd.append(pause_features)
    X_pause = np.concatenate((all_counts_cc, all_counts_cd), axis=0).astype(np.float32)
    y_reg_cc = utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
    y_reg_cd = utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))
    y_reg_pause = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
    X_reg_pause = np.copy(X_pause)
    y_cc = np.zeros((len(all_counts_cc), 2))
    y_cc[:, 0] = 1
    y_cd = np.zeros((len(all_counts_cd), 2))
    y_cd[:, 1] = 1
    y_pause = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
    filenames_pause = np.concatenate((cc_transcription_files, cd_transcription_files), axis=0)
    cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/cc/*.csv')))
    X_cc = np.array([get_compare_features(f) for f in cc_files])
    y_cc = np.zeros((X_cc.shape[0], 2))
    y_cc[:, 0] = 1
    cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/cd/*.csv')))
    X_cd = np.array([get_compare_features(f) for f in cd_files])
    y_cd = np.zeros((X_cd.shape[0], 2))
    y_cd[:, 1] = 1
    X_compare = np.concatenate((X_cc, X_cd), axis=0).astype(np.float32)
    y_compare = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
    X_reg_compare = np.copy(X_compare)
    y_reg_cc = utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
    y_reg_cd = utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))
    y_reg_compare = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
    filenames_compare = np.concatenate((cc_files, cd_files), axis=0)
    #assert np.array_equal(y_intervention, y_pause) and X_intervention.shape[0] == X_pause.shape[0], '~ Data streams are different ~'
    print('~ Data streams verified ~')
    p = np.random.permutation(X_intervention.shape[0])
    X_intervention = X_intervention[p]
    X_pause = X_pause[p]
    y_pause = y_pause[p]
    if p < 108:
        X_pause = X_pause[p]
    else:
        X_pause = Y_pause[p]
    X_compare = X_compare[p]
    y_intervention = y_intervention[p]
    y_compare = y_compare[p]
    y_reg_intervention = y_reg_intervention[p]
    y_reg_pause = y_reg_pause[p]
    y_reg_compare = y_reg_compare[p]
    subjects = []
    return {
        'intervention': X_intervention,
        'pause': X_pause,
        'compare': X_compare,
        'y_clf': y_intervention,
        'y_reg': y_reg_intervention,
        'subjects': subjects
    }
print(prepare_data(dataset_dir, config))
def prepare_data(dataset_dir, config):
    def get_features(files, feature_function):
        features = []
        for file in files:
            features.append(feature_function(file, config.longest_speaker_length))
        return np.array(features)

    def get_classification_values(filename):
        # Implement your logic to retrieve classification values from the file
        pass

    def get_regression_values(filename):
        # Implement your logic to retrieve regression values from the file
        pass

    cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))
    all_speakers_cc = get_features(cc_files, get_intervention_features)

    cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))
    all_speakers_cd = get_features(cd_files, get_intervention_features)

    y_cc = np.zeros((len(all_speakers_cc), 2))
    y_cc[:, 0] = 1

    y_cd = np.zeros((len(all_speakers_cd), 2))
    y_cd[:, 1] = 1

    X_intervention = np.concatenate((all_speakers_cc, all_speakers_cd), axis=0).astype(np.float32)
    y_intervention = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
    filenames_intervention = np.concatenate((cc_files, cd_files), axis=0)

    y_reg_cc = get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
    y_reg_cd = get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))
    if y_reg_cc.ndim == 0:
        y_reg_cc = np.expand_dims(y_reg_cc, axis=0)

    if y_reg_cd.ndim == 0:
        y_reg_cd = np.expand_dims(y_reg_cd, axis=0)
    y_reg_intervention = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
    X_reg_intervention = np.copy(X_intervention)

    cc_transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))
    cc_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.wav')))
    all_counts_cc = get_features(zip(cc_transcription_files, cc_audio_files), get_pause_features)

    cd_transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))
    cd_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.wav')))
    all_counts_cd = get_features(zip(cd_transcription_files, cd_audio_files), get_pause_features)

    X_pause = np.concatenate((all_counts_cc, all_counts_cd), axis=0).astype(np.float32)
    y_reg_cc = get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
    y_reg_cd = get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))
    y_reg_pause = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
    X_reg_pause = np.copy(X_pause)

    y_cc = np.zeros((len(all_counts_cc), 2))
    y_cc[:, 0] = 1

    y_cd = np.zeros((len(all_counts_cd), 2))
    y_cd[:, 1] = 1

    y_pause = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
    filenames_pause = np.concatenate((cc_transcription_files, cd_transcription_files), axis=0)

    cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/cc/*.csv')))
    X_cc = get_features(cc_files, get_compare_features)
    y_cc = np.zeros((X_cc.shape[0], 2))
    y_cc[:, 0] = 1

    cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/cd/*.csv')))
    X_cd = get_features(cd_files, get_compare_features)
    y_cd = np.zeros((X_cd.shape[0], 2))
    y_cd[:, 1] = 1

    X_compare = np.concatenate((X_cc, X_cd), axis=0).astype(np.float32)
    y_compare = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
    X_reg_compare = np.copy(X_compare)

    y_reg_cc = get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
    y_reg_cd = get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

    if y_reg_cc is not None and y_reg_cc.ndim == 0:
        y_reg_cc = np.expand_dims(y_reg_cc, axis=0)

    if y_reg_cd is not None and y_reg_cd.ndim == 0:
        y_reg_cd = np.expand_dims(y_reg_cd, axis=0)

    y_reg_intervention = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
    filenames_compare = np.concatenate((cc_files, cd_files), axis=0)

    print('~ Data streams verified ~')

    p = np.random.permutation(X_intervention.shape[0])

    X_intervention = X_intervention[p]
    X_pause = X_pause[p]
    y_pause = y_pause[p]

    if p < 108:
        X_pause = X_pause[p]
    else:
        X_pause = Y_pause[p]

    X_compare = X_compare[p]
    y_intervention = y_intervention[p]
    y_compare = y_compare[p]
    y_reg_intervention = y_reg_intervention[p]
    y_reg_pause = y_reg_pause[p]
    y_reg_compare = y_reg_compare[p]
    subjects = []

    return {
        'intervention': X_intervention,
        'pause': X_pause,
        'compare': X_compare,
        'y_clf': y_intervention,
        'y_reg': y_reg_intervention,
        'subjects': subjects
    }

print(prepare_data(dataset_dir, config))


def read_wave(path):

    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()

        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)

        converted = audioop.ratecv(data, sample_width, num_channels, sample_rate, 32000, None)
        return converted[0], 32000

class Frame(object):


    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):

    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):

    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    silenced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if is_speech:
            silenced_frames.append(5)
        else:
            silenced_frames.append(10)

    return silenced_frames



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

    return segment_intervals

def prepare_data_new(dataset_dir, config):

    subject_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.wav')) +
                           glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.wav')))
    subjects = np.array(sorted(list(set([os.path.splitext(os.path.basename(file))[0] for file in subject_files]))))
    print(subjects)

    if 'ADReSS' in dataset_dir:
        cc_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.mp3')))
    else:
        cc_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.wav')))

    max_len = 800

    all_counts_cc = []
    for audio_file in cc_audio_files:
        print(audio_file, "af")
        pause_features = get_pause_masks(audio_file)
        all_counts_cc.append(pause_features[:max_len])

    if dataset_dir == 'D:/projects/dmentia project/ADReSS-IS2020-train/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd':
        cd_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.mp3')))
    else:
        cd_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.wav')))

    all_counts_cd = []
    for audio_file in cd_audio_files:
        pause_features = get_pause_masks(audio_file)
        all_counts_cd.append(pause_features[:max_len])

    all_counts_cd = np.asarray(all_counts_cd)
    all_counts_cc = np.asarray(all_counts_cc)

    print("all_counts_cc:", all_counts_cc.shape)
    print("all_counts_cd:", all_counts_cd.shape)

    X_pause = np.concatenate((all_counts_cc, all_counts_cd), axis=0).astype(np.float32)

    print("==>> all_counts_cc:", all_counts_cc)
    print("==>> all_counts_cd:", all_counts_cd)

    y_reg_cc = utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
    y_reg_cd = utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

    y_reg_pause = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)

    X_reg_pause = np.copy(X_pause)

    y_cc = np.zeros((len(all_counts_cc), 2))
    y_cc[:, 0] = 1

    y_cd = np.zeros((len(all_counts_cd), 2))
    y_cd[:, 1] = 1

    y_pause = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)

    print("X_pause shape:", X_pause.shape)
    print("y_pause shape:", y_pause.shape)
    print("y_reg_pause shape:", y_reg_pause.shape)

    return {
        'silences': X_pause,
        'y_clf': y_pause,
        'y_reg': y_reg_pause,
        'subjects': subjects
    }

dataset_dir = r'D:\projects\dmentia project\ADReSS-IS2020-test\ADReSS-IS2020-data'

def prepare_test_data(dataset_dir, config):


    ################################## SUBJECTS ################################

    subject_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/*.cha')))

    # Check if the subject_files list is empty
    if not subject_files:
        print("No subject files found.")
        return None

    subjects = []  # Initialize the list to store the subject names
    for file_path in subject_files:
        # Extract the subject name using regex pattern matching
        match = re.search(r"\\S(\d+)\.cha$", file_path)
        if match:
            subject_number = match.group(1)
            subjects.append(f"S{subject_number}")
        else:
            print(f"Invalid file name format: {file_path}")


    # Check if the subjects list is empty
    if not subjects:
        print("No valid subjects found.")
        return None

    subjects = np.array(sorted(list(set(subjects))))



    ################################## INTERVENTION ####################################

    transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/*.cha')))
    all_speakers = []
    for filename in transcription_files:
        all_speakers.append(get_intervention_features(filename, config.longest_speaker_length))
    X_intervention = np.array(all_speakers)

    ################################## PAUSE ####################################

    audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/*.wav')))

    all_counts = []
    for t_f, a_f in zip(transcription_files, audio_files):
        pause_features = get_pause_features(t_f, a_f)
        all_counts.append(pause_features)
    X_pause = np.array(all_counts)

    ################################## COMPARE ####################################

    compare_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/*.csv')))
    X_compare = np.array([get_compare_features(f) for f in compare_files])

    metadata_filename = os.path.join(dataset_dir, 'meta_data.txt')
    if not os.path.exists(metadata_filename):
        print(f"File not found: {metadata_filename}")
        # Handle the missing file error
        y = None
        y_reg = None
    else:
        y = utils.get_classification_values(metadata_filename)
        y_reg = utils.get_regression_values(metadata_filename)

    assert X_intervention.shape[0] == X_pause.shape[0], '~ Data streams are different ~'
    print('~ Data streams verified ~')

    return {
        'intervention': X_intervention,
        'pause': X_pause,
        'compare': X_compare,
        'y_clf': y,
        'y_reg': y_reg,
        'subjects': subjects
    }

print(prepare_test_data(dataset_dir, config))'''
import glob
import os
import math
import time
import re
import csv
from config import config
import numpy as np

np.random.seed(0)
p = np.random.permutation(108)  # n_samples = 108
p_subjects = np.random.RandomState(seed=0).permutation(242)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import sys
import utils

dataset_dir=r'D:\projects\dmentia project\ADReSS-IS2020-train\ADReSS-IS2020-data\train'

def get_pause_features(transcription_filename, audio_filename, audio_length_normalization=10):
    '''
    Pause features include word rate, pause rate of various kinds of pauses and utterances, and intervention rate
    '''
    audio_len = utils.get_audio_length(audio_filename) / audio_length_normalization

    with open(transcription_filename, 'r',encoding = 'utf-8') as f:
        content = f.read()
        word_rate = utils.words_count(content) / (50 * audio_len)
        pause_rates = utils.get_pauses_cnt(content) / audio_len
        inv_rate = utils.get_n_interventions(content) / audio_len

    pause_features = np.concatenate(([inv_rate], pause_rates, [word_rate]), axis=-1)

    return pause_features
def get_intervention_features(transcription_filename, max_length=40):
    '''
    Intervention features include one hot encoded sequence of speaker (par or inv) in the conversation
    '''
    speaker_dict = {
        'INV': [0, 0, 1],
        'PAR': [0, 1, 0],
        'padding': [1, 0, 0]
    }

    with open(transcription_filename, 'r',encoding='utf-8') as f:
        content = f.read()
        content = content.split('\n')
        speakers = []

        for c in content:
            if 'INV' in c:
                speakers.append('INV')
            if 'PAR' in c:
                speakers.append('PAR')

        PAR_first_index = speakers.index('PAR')
        PAR_last_index = len(speakers) - speakers[::-1].index('PAR') - 1
        intervention_features = speakers[PAR_first_index:PAR_last_index]

    intervention_features = list(map(lambda x: speaker_dict[x], intervention_features))

    if len(intervention_features) > max_length:
        intervention_features = intervention_features[:max_length]
    else:
        pad_length = max_length - len(intervention_features)
        intervention_features = intervention_features + [speaker_dict['padding']] * pad_length

    return intervention_features
def get_spectogram_features(spectogram_filename):
    '''
    Spectogram features include MFCC which has been pregenerated for the audio file
    '''
    mel = np.load(spectogram_filename)
    # mel = feature_normalize(mel)
    mel = np.expand_dims(mel, axis=-1)
    return mel
def get_compare_features(compare_filename):
    compare_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/*.csv')))
    compare_features = []
    max_rows = 50
    with open(compare_filename, 'r') as file:
        content = csv.reader(file)
        rows =[]
        for i ,row in enumerate(content):
            if i >= max_rows:
                break
            rows.append(row)
        for row in content:
            compare_features = row

    compare_features_floats = [float(item) for item in compare_features[1:-1]]
    return compare_features_floats

import numpy as np
import types

import glob
import os
import numpy as np
import types
import utils

def prepare_data(dataset_dir, config):

    cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))
    all_speakers_cc = []
    for filename in cc_files:
        all_speakers_cc.append(get_intervention_features(filename, config.longest_speaker_length))

    cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))
    all_speakers_cd = []
    for filename in cd_files:
        all_speakers_cd.append(get_intervention_features(filename, config.longest_speaker_length))

    y_cc = np.zeros((len(all_speakers_cc), 2))
    y_cc[:, 0] = 1

    y_cd = np.zeros((len(all_speakers_cd), 2))
    y_cd[:, 1] = 1

    X_intervention = np.concatenate((all_speakers_cc, all_speakers_cd), axis=0).astype(np.float32)
    y_intervention = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
    filenames_intervention = np.concatenate((cc_files, cd_files), axis=0)

    y_reg_cc = utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
    y_reg_cd = utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

    y_reg_intervention = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)
    X_reg_intervention = np.copy(X_intervention)
    cc_transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cc/*.cha')))
    cc_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.wav')))

    all_counts_cc = []
    for t_f, a_f in zip(cc_transcription_files, cc_audio_files):
        pause_features = get_pause_features(t_f, a_f)
        all_counts_cc.append(pause_features)

    cd_transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/cd/*.cha')))
    cd_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.wav')))

    all_counts_cd = []
    for t_f, a_f in zip(cd_transcription_files, cd_audio_files):
        pause_features = get_pause_features(t_f, a_f)
        all_counts_cd.append(pause_features)

    X_pause = np.concatenate((all_counts_cc, all_counts_cd), axis=0).astype(np.float32)

    y_reg_cc = utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
    y_reg_cd = utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

    y_reg_pause = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)

    X_reg_pause = np.copy(X_pause)

    y_cc = np.zeros((len(all_counts_cc), 2))
    y_cc[:, 0] = 1

    y_cd = np.zeros((len(all_counts_cd), 2))
    y_cd[:, 1] = 1

    y_pause = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)
    filenames_pause = np.concatenate((cc_transcription_files, cd_transcription_files), axis=0)

    cc_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/cc/*.csv')))
    X_cc = np.array([get_compare_features(f) for f in cc_files])
    y_cc = np.zeros((X_cc.shape[0], 2))
    y_cc[:, 0] = 1

    cd_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/cd/*.csv')))
    X_cd = np.array([get_compare_features(f) for f in cd_files])
    y_cd = np.zeros((X_cd.shape[0], 2))
    y_cd[:, 1] = 1
    X_compare = np.concatenate((X_cc, X_cd), axis=0).astype(np.float32)

    y_compare = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)

    X_reg_compare = np.copy(X_compare)

    y_reg_cc = utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
    y_reg_cd = utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

    y_reg_compare = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)

    filenames_compare = np.concatenate((cc_files, cd_files), axis=0)
    print("X_intervention shape:", X_intervention.shape)
    print("X_pause shape:", X_pause.shape)
    print("y_intervention shape:", y_intervention.shape)
    print("y_pause shape:", y_pause.shape)

    # Verify data streams
    assert np.array_equal(y_intervention, y_pause) and X_intervention.shape[0] == X_pause.shape[0], '~ Data streams are different ~'
    print('~ Data streams verified ~')

    # Perform permutation
    p = np.random.permutation(X_intervention.shape[0])
    X_intervention = X_intervention[p]
    X_pause = X_pause[p]
    X_compare = X_compare[p]
    y_intervention = y_intervention[p]
    y_pause = y_pause[p]
    y_compare = y_compare[p]
    y_reg_intervention = y_reg_intervention[p]
    y_reg_pause = y_reg_pause[p]
    y_reg_compare = y_reg_compare[p]

    subjects = []  # Define and populate the subjects list

    return {
        'intervention': X_intervention,
        'pause': X_pause,
        'compare': X_compare,
        'y_clf': y_intervention,
        'y_reg': y_reg_intervention,
        'subjects': subjects
    }
print(prepare_data(dataset_dir, config))
import collections
import contextlib
import sys
import wave

import webrtcvad
import audioop

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()

        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)

        converted = audioop.ratecv(data, sample_width, num_channels, sample_rate, 32000, None)
        return converted[0], 32000

class Frame(object):
    """Represents a "frame" of audio data."""

    def _init_(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    silenced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if is_speech:
            silenced_frames.append(5)
        else:
            silenced_frames.append(10)

    return silenced_frames


def get_pause_masks(file):
    frame_duration_ms = 30

    audio, sample_rate = read_wave(file)
    vad = webrtcvad.Vad()
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, frame_duration_ms, 10, vad, frames)

    segments = np.asarray(segments)
    # segments = (segments - np.mean(segments))/np.std(segments)
    # print(segments)
    return segments

def prepare_data_new(dataset_dir, config):
    '''
    Prepare all data
    '''
    subject_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.wav')) +
                           glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.wav')))
    subjects = np.array(sorted(list(set([os.path.splitext(os.path.basename(file))[0] for file in subject_files]))))
    print(subjects)

    if 'ADReSS' in dataset_dir:
        cc_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.mp3')))
    else:
        cc_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cc/*.wav')))

    max_len = 800

    all_counts_cc = []
    for audio_file in cc_audio_files:
        print(audio_file, "af")
        pause_features = get_pause_masks(audio_file)
        all_counts_cc.append(pause_features[:max_len])

    if dataset_dir == 'D:/projects/dmentia project/ADReSS-IS2020-train/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd':
        cd_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.mp3')))
    else:
        cd_audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/cd/*.wav')))

    all_counts_cd = []
    for audio_file in cd_audio_files:
        pause_features = get_pause_masks(audio_file)
        all_counts_cd.append(pause_features[:max_len])

    all_counts_cd = np.asarray(all_counts_cd)
    all_counts_cc = np.asarray(all_counts_cc)

    print("all_counts_cc:", all_counts_cc.shape)
    print("all_counts_cd:", all_counts_cd.shape)

    X_pause = np.concatenate((all_counts_cc, all_counts_cd), axis=0).astype(np.float32)

    print("==>> all_counts_cc:", all_counts_cc)
    print("==>> all_counts_cd:", all_counts_cd)

    y_reg_cc = utils.get_regression_values(os.path.join(dataset_dir, 'cc_meta_data.txt'))
    y_reg_cd = utils.get_regression_values(os.path.join(dataset_dir, 'cd_meta_data.txt'))

    y_reg_pause = np.concatenate((y_reg_cc, y_reg_cd), axis=0).astype(np.float32)

    X_reg_pause = np.copy(X_pause)

    y_cc = np.zeros((len(all_counts_cc), 2))
    y_cc[:, 0] = 1

    y_cd = np.zeros((len(all_counts_cd), 2))
    y_cd[:, 1] = 1

    y_pause = np.concatenate((y_cc, y_cd), axis=0).astype(np.float32)

    print("X_pause shape:", X_pause.shape)
    print("y_pause shape:", y_pause.shape)
    print("y_reg_pause shape:", y_reg_pause.shape)

    return {
        'silences': X_pause,
        'y_clf': y_pause,
        'y_reg': y_reg_pause,
        'subjects': subjects
    }

dataset_dir = r'D:\projects\dmentia project\ADReSS-IS2020-test\ADReSS-IS2020-data\test'

import glob
import numpy as np
import os
import re

import glob
import numpy as np
import os
import re

def prepare_test_data(dataset_dir, config):
    '''
    Prepare test data
    '''

    ################################## SUBJECTS ################################

    subject_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/*.cha')))

    # Check if the subject_files list is empty
    if not subject_files:
        print("No subject files found.")
        return None

    subjects = []  # Initialize the list to store the subject names
    for file_path in subject_files:
        # Extract the subject name using regex pattern matching
        match = re.search(r"\\S(\d+)\.cha$", file_path)
        if match:
            subject_number = match.group(1)
            subjects.append(f"S{subject_number}")
        else:
            print(f"Invalid file name format: {file_path}")


    # Check if the subjects list is empty
    if not subjects:
        print("No valid subjects found.")
        return None

    subjects = np.array(sorted(list(set(subjects))))

    ######################################################################

    ################################## INTERVENTION ####################################

    transcription_files = sorted(glob.glob(os.path.join(dataset_dir, 'transcription/*.cha')))
    all_speakers = []
    for filename in transcription_files:
        all_speakers.append(get_intervention_features(filename, config.longest_speaker_length))
    X_intervention = np.array(all_speakers)

    ################################## PAUSE ####################################

    audio_files = sorted(glob.glob(os.path.join(dataset_dir, 'Full_wave_enhanced_audio/*.wav')))

    all_counts = []
    for t_f, a_f in zip(transcription_files, audio_files):
        pause_features = get_pause_features(t_f, a_f)
        all_counts.append(pause_features)
    X_pause = np.array(all_counts)

    ################################## COMPARE ####################################

    compare_files = sorted(glob.glob(os.path.join(dataset_dir, 'compare/*.csv')))
    X_compare = np.array([get_compare_features(f) for f in compare_files])

    metadata_filename = os.path.join(dataset_dir, 'meta_data.txt')
    if not os.path.exists(metadata_filename):
        print(f"File not found: {metadata_filename}")
        # Handle the missing file error
        y = None
        y_reg = None
    else:
        y = utils.get_classification_values(metadata_filename)
        y_reg = utils.get_regression_values(metadata_filename)

    assert X_intervention.shape[0] == X_pause.shape[0], '~ Data streams are different ~'
    print('~ Data streams verified ~')

    return {
        'intervention': X_intervention,
        'pause': X_pause,
        'compare': X_compare,
        'y_clf': y,
        'y_reg': y_reg,
        'subjects': subjects
    }

print(prepare_test_data(dataset_dir, config))