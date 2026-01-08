import os
import numpy as np
import pandas as pd
import librosa
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

lyrics_model = None  

def extract_mel_spectrogram(audio_path, sr=22050, n_mels=64, hop_length=512, max_len=64):
    y, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
  
    if S_dB.shape[1] < max_len:
        pad_width = max_len - S_dB.shape[1]
        S_dB = np.pad(S_dB, ((0, 0), (0, pad_width)), mode='constant')
    else:
        S_dB = S_dB[:, :max_len]

    S_norm = (S_dB + 80) / 80
    S_norm = S_norm.astype(np.float32)
    return np.expand_dims(S_norm, axis=0)

def extract_lyrics_embedding(lyrics_path):
    global lyrics_model
    if lyrics_model is None:
        from sentence_transformers import SentenceTransformer
        lyrics_model_local = SentenceTransformer('all-MiniLM-L6-v2')
        lyrics_model = lyrics_model_local

    with open(lyrics_path, 'r', encoding='utf-8') as f:
        text = f.read()
    embedding = lyrics_model.encode(text)
    return embedding.astype(np.float32)

def load_multimodal_features(csv_path, audio_base_dir="data/audio", lyrics_base_dir="data/lyrics", max_len=64, resolve_paths_only=False):
    df = pd.read_csv(csv_path)

    audio_features = []
    lyrics_features = []
    valid_rows = []

    print("Extracting multi-modal features...")

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
        raw_lyrics = str(row['lyrics_path'])

        audio_path = _find_in_dir(audio_base_dir, raw_audio)
        lyrics_path = _find_in_dir(lyrics_base_dir, raw_lyrics)

        if audio_path is None or lyrics_path is None:
            continue

        if resolve_paths_only:
            audio_features.append(audio_path)
            lyrics_features.append(lyrics_path)
            valid_rows.append(row.to_dict())
            continue

        mel_spec = extract_mel_spectrogram(audio_path, max_len=max_len)
        lyrics_emb = extract_lyrics_embedding(lyrics_path)

        audio_features.append(mel_spec)
        lyrics_features.append(lyrics_emb)
        valid_rows.append(row.to_dict())

    if len(audio_features) == 0:
        print("Warning: No valid audio-lyrics pairs found!")


    if len(audio_features) > 0:
        audio_features = np.stack(audio_features)  # shape (N,1,n_mels,max_len)
    else:
        audio_features = np.empty((0, 1, 64, max_len), dtype=np.float32)

    if len(lyrics_features) > 0:
        lyrics_features = np.stack(lyrics_features)
    else:
        lyrics_features = np.empty((0,), dtype=np.float32)

    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    return audio_features, lyrics_features, valid_df
