import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling
from datasets import load_dataset
from f5.ogf5tts import OGF5TTSService
from f5_tts.infer.utils_infer import infer_process
import random
import tempfile
import os
import torchaudio
import pickle
from tqdm import tqdm
import numpy as np
from pathlib import Path
from f5.utils_infer import load_vocoder
import torchaudio.functional as AF
import librosa

# Configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATASET_PATH = "./generated_dataset.pkl"
DATASET_SIZE = 150
BATCH_SIZE = 8  # Process texts in batches
MAX_TEXT_LENGTH = 150  # Maximum length for each text sample
SAMPLE_RATE = 24000
N_MELS = 80
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024
DATASET_VERSION = 2  # Increment version since we're changing dataset source

def load_existing_dataset():
    """Load existing dataset if available"""
    if os.path.exists(DATASET_PATH):
        print(f"Loading existing dataset from {DATASET_PATH}")
        with open(DATASET_PATH, 'rb') as f:
            try:
                data = pickle.load(f)
                if isinstance(data, dict) and 'data' in data:
                    dataset = data['data']
                else:
                    dataset = data  # Handle old format
                print(f"Found {len(dataset)} existing samples")
                return dataset
            except Exception as e:
                print(f"Error loading dataset: {str(e)}")
                return []
    print("No existing dataset found, starting fresh")
    return []

def prepare_dolly():
    """Load and prepare Dolly dataset"""
    print("Loading Dolly-15k dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    # Extract and clean response texts
    texts = []
    for item in dataset:
        text = item['response'].strip()
        # Filter out too short or too long responses
        if 50 < len(text) < MAX_TEXT_LENGTH:
            texts.append(text)
    
    # Randomly sample required number of texts
    if len(texts) > DATASET_SIZE:
        texts = random.sample(texts, DATASET_SIZE)
    
    print(f"Prepared {len(texts)} text samples")
    return texts

def generate_embeddings(texts, existing_texts=None):
    """Generate LLaMA embeddings for texts"""
    processed_texts = []
    
    try:
        llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3b").to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b")
        tokenizer.pad_token = tokenizer.eos_token
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            
            # Process each text individually to avoid padding issues
            for text in batch_texts:
                if existing_texts and text in existing_texts:
                    continue
                    
                with torch.no_grad():
                    # Tokenize without padding
                    inputs = tokenizer(
                        text,
                        padding=False,
                        truncation=True,
                        max_length=MAX_TEXT_LENGTH,
                        return_tensors="pt"
                    ).to(DEVICE)
                    
                    outputs = llama_model(**inputs, output_hidden_states=True)
                    embedding = outputs.hidden_states[-1].cpu().squeeze(0)  # Remove batch dimension
                    
                    processed_texts.append({
                        'text': text,
                        'embeddings': embedding,
                        'version': DATASET_VERSION
                    })
                    
    finally:
        del llama_model
        torch.cuda.empty_cache()
    
    return processed_texts

def extract_prosody_features(mel_spec, audio):
    """Extract detailed prosody features from mel spectrogram and audio"""
    features = {}
    
    # Energy contour (from mel spectrogram)
    features['energy_contour'] = torch.norm(mel_spec, dim=1)
    
    # Convert audio to numpy for librosa
    if torch.is_tensor(audio):
        audio = audio.cpu().numpy()
    
    # Pitch contour (using librosa)
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=SAMPLE_RATE
    )
    features['f0_contour'] = torch.from_numpy(f0).float()
    features['voiced_mask'] = torch.from_numpy(voiced_flag).float()
    
    # Spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE)
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)
    
    features['spectral_centroid'] = torch.from_numpy(spec_centroid).float()
    features['spectral_bandwidth'] = torch.from_numpy(spec_bandwidth).float()
    features['spectral_rolloff'] = torch.from_numpy(spec_rolloff).float()
    
    # Rhythm features (onset strength)
    onset_env = librosa.onset.onset_strength(y=audio, sr=SAMPLE_RATE)
    features['onset_strength'] = torch.from_numpy(onset_env).float()
    
    # Duration features
    tempo, beats = librosa.beat.beat_track(y=audio, sr=SAMPLE_RATE)
    features['tempo'] = torch.tensor(tempo).float()
    features['beat_frames'] = torch.from_numpy(beats).float()
    
    return features

def extract_alignment_info(mel_spec, text_tokens, hidden_states):
    """Extract alignment information between text and audio"""
    alignment_info = {}
    
    # Cross-attention weights between text and mel frames
    text_len = len(text_tokens)
    mel_len = mel_spec.shape[0]
    
    # Simple monotonic alignment matrix
    alignment_matrix = torch.zeros(text_len, mel_len)
    frames_per_token = mel_len / text_len
    
    for i in range(text_len):
        start_frame = int(i * frames_per_token)
        end_frame = int((i + 1) * frames_per_token)
        alignment_matrix[i, start_frame:end_frame] = 1.0
    
    alignment_info['attention_matrix'] = alignment_matrix
    
    # Token-level durations
    durations = torch.tensor([int(frames_per_token)] * text_len)
    alignment_info['token_durations'] = durations
    
    # Hidden state transitions
    if hidden_states is not None:
        transitions = F.cosine_similarity(
            hidden_states[:-1],
            hidden_states[1:],
            dim=-1
        )
        alignment_info['hidden_transitions'] = transitions
    
    return alignment_info

def generate_speech(processed_texts, existing_dataset):
    """Generate speech with enhanced feature extraction"""
    if not processed_texts:
        print("No texts to process for speech")
        return existing_dataset
    
    dataset = existing_dataset.copy()
    
    print("\nLoading F5-TTS model...")
    # Load vocoder once
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False)
    f5_service = OGF5TTSService(model_dir="weights", voice_profile="Bob", vocoder=vocoder)
    
    # Save reference audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as ref_temp:
        torchaudio.save(ref_temp.name, f5_service.ref_audio.cpu(), 24000)
        ref_audio_path = ref_temp.name
    
    try:
        print("\nGenerating speech...")
        pbar = tqdm(total=len(processed_texts))
        
        for item in processed_texts:
            try:
                print(f"gen_text {len(dataset)} {item['text']}")
                # Generate mel spectrogram and audio
                audio_output, _, mel_spec = infer_process(
                    ref_audio=ref_audio_path,
                    ref_text=f5_service.ref_text,
                    gen_text=item['text'],
                    model_obj=f5_service.model,
                    vocoder=f5_service.vocoder,
                    mel_spec_type="vocos",
                    speed=1.0,
                    nfe_step=32,
                    cfg_strength=2.0,
                    sway_sampling_coef=-1.0,
                    device=DEVICE
                )
                
                # Extract prosody features
                prosody_features = extract_prosody_features(
                    torch.from_numpy(mel_spec),
                    audio_output
                )
                
                # Use text length directly for alignment
                text_length = len(item['text'].split())  # Simple word-based tokenization
                
                # Extract alignment information without tokens
                alignment_info = extract_alignment_info(
                    torch.from_numpy(mel_spec),
                    range(text_length),  # Use word positions instead of tokens
                    item['embeddings']
                )
                
                # Add to dataset with enhanced features
                dataset.append({
                    'embeddings': item['embeddings'],
                    'mel_spec': torch.from_numpy(mel_spec).cpu(),
                    'text': item['text'],
                    'prosody': prosody_features,
                    'alignment': alignment_info,
                    'audio': torch.from_numpy(audio_output).cpu(),
                    'text_length': text_length  # Store word count for masking
                })
                
                pbar.update(1)
                
                # Save progress periodically
                if len(dataset) % 10 == 0:
                    with open(DATASET_PATH, 'wb') as f:
                        pickle.dump({
                            'data': dataset,
                            'metadata': {
                                'sample_rate': SAMPLE_RATE,
                                'n_mels': N_MELS,
                                'hop_length': HOP_LENGTH,
                                'win_length': WIN_LENGTH,
                                'n_fft': N_FFT
                            }
                        }, f)
                    print(f"\nSaved progress: {len(dataset)} samples")
                    
            except Exception as e:
                print(f"\nError processing text: {str(e)}")
                continue
    
    finally:
        # Clean up
        if os.path.exists(ref_audio_path):
            os.unlink(ref_audio_path)
    
    return dataset

def generate_dataset():
    # Load existing dataset
    dataset = load_existing_dataset()
    if dataset:
        stored_version = getattr(dataset[0], 'version', 1)
        if stored_version < DATASET_VERSION:
            print(f"Old dataset version {stored_version} detected. Regenerating...")
            dataset = []
        
    existing_texts = {item['text'] for item in dataset}
    
    if len(dataset) >= DATASET_SIZE:
        print(f"Dataset already contains {len(dataset)} samples. No additional generation needed.")
        return
    
    # Load and prepare Dolly dataset instead of WikiText
    dolly_texts = prepare_dolly()
    
    # Generate embeddings
    processed_texts = generate_embeddings(dolly_texts, existing_texts)
    
    if not processed_texts:
        print("No new texts to process")
        return
    
    # Generate speech
    dataset = generate_speech(processed_texts, dataset)
    
    # Save final dataset with metadata
    print("\nSaving final dataset...")
    with open(DATASET_PATH, 'wb') as f:
        pickle.dump({
            'data': dataset,
            'metadata': {
                'sample_rate': SAMPLE_RATE,
                'n_mels': N_MELS,
                'hop_length': HOP_LENGTH,
                'win_length': WIN_LENGTH,
                'n_fft': N_FFT,
                'source': 'dolly-15k'  # Add source information
            }
        }, f)
    
    print(f"Completed! Total samples in dataset: {len(dataset)}")

if __name__ == "__main__":
    generate_dataset() 