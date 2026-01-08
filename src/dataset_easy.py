import librosa
import numpy as np
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd

lyrics_model = None  

def extract_mel_spectrogram(audio_path, sr=22050, n_mels=64, hop_length=512, max_len=64):
    y, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    if S_dB.shape[1] < max_len:
        pad_width = max_len - S_dB.shape[1]
        S_dB = np.pad(S_dB, ((0,0),(0,pad_width)), mode='constant')
    else:
        S_dB = S_dB[:, :max_len]
    S_norm = (S_dB + 80) / 80
    S_norm = np.clip(S_norm, 0.0, 1.0).astype(np.float32)
    return S_norm

def extract_lyrics_embedding(lyrics_path):
    global lyrics_model
    if lyrics_model is None:
        from sentence_transformers import SentenceTransformer
        lyrics_model_local = SentenceTransformer('all-MiniLM-L6-v2')
        lyrics_model = lyrics_model_local

    with open(lyrics_path, 'r', encoding='utf-8') as f:
        text = f.read()
    embedding = lyrics_model.encode(text)
    return embedding

def load_multimodal_features(csv_path):
    df = pd.read_csv(csv_path)
    audio_features = []
    lyrics_features = []
    valid_rows = []
    print("Extracting multi-modal features...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row['audio_path']
        lyrics_path = row['lyrics_path']
        if not os.path.exists(audio_path) or not os.path.exists(lyrics_path):
            continue
        mel_spec = extract_mel_spectrogram(audio_path)
        audio_features.append(mel_spec)
        lyrics_emb = extract_lyrics_embedding(lyrics_path)
        lyrics_features.append(lyrics_emb)
        valid_rows.append(row)
    audio_features = np.array(audio_features)
    lyrics_features = np.array(lyrics_features)
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    return audio_features, lyrics_features, valid_df


def load_audio_features(csv_path, audio_base_dir="data/audio", max_len=64):
    df = pd.read_csv(csv_path)
    audio_features = []
    valid_rows = []

    def _find_in_dir(base_dir, candidate_path):
        if os.path.isabs(candidate_path) and os.path.exists(candidate_path):
            return candidate_path
        if os.path.exists(candidate_path):
            return candidate_path
        basename = os.path.basename(candidate_path)
        for root, _, files in os.walk(base_dir):
            if basename in files:
                return os.path.join(root, basename)
        return None

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        raw_audio = str(row['audio_path'])
        audio_path = _find_in_dir(audio_base_dir, raw_audio)
        if audio_path is None:
            continue
        mel_spec = extract_mel_spectrogram(audio_path, max_len=max_len)
        mel_flat = mel_spec.flatten().astype(np.float32)
        audio_features.append(mel_flat)
        valid_rows.append(row.to_dict())

    if len(audio_features) == 0:
        return np.empty((0, 64 * max_len), dtype=np.float32), pd.DataFrame(valid_rows)

    filtered = []
    filtered_rows = []
    for x, r in zip(audio_features, valid_rows):
        if np.isfinite(x).all():
            filtered.append(x)
            filtered_rows.append(r)

    if len(filtered) == 0:
        return np.empty((0, 64 * max_len), dtype=np.float32), pd.DataFrame(filtered_rows)

    X = np.stack(filtered)
    valid_df = pd.DataFrame(filtered_rows).reset_index(drop=True)
    return X, valid_df
