import torch
import torchaudio
import sounddevice as sd
import os
import soundfile as sf
import tempfile
from types import SimpleNamespace
import numpy as np
from pydub import AudioSegment
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from f5_tts.model import DiT, CFM
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    target_sample_rate,
)
from f5_tts.model.modules import MelSpec

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_path = os.path.join("weights", "final_finetuned_model.pt")
vocab_path = os.path.join("vocab.txt")
VOICE_PROFILE_DIR = "voice_profiles"

# Load vocabulary
def load_vocab(vocab_file):
    """Load vocabulary from file and create char map"""
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f.readlines()]
    
    vocab_char_map = {}
    for i, char in enumerate(vocab):
        if char:  # Skip empty lines
            vocab_char_map[char] = i
            
    return vocab_char_map, len(vocab_char_map) + 1

# Initialize model
vocab_char_map, vocab_size = load_vocab(vocab_path)


# Initialize model
model_cfg = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    conv_layers=4
)

mel_spec_kwargs = dict(
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    mel_spec_type="vocos"
)

# Create CFM model
model = CFM(
    transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
    mel_spec_kwargs=mel_spec_kwargs,
    vocab_char_map=vocab_char_map,
)

# Load model and vocoder
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
vocoder = load_vocoder(vocoder_name="vocos", is_local=False)


class VoiceProfile:
    def __init__(self, profile_name):
        self.profile_name = profile_name
        self.profile_dir = os.path.join(VOICE_PROFILE_DIR, profile_name)
        os.makedirs(self.profile_dir, exist_ok=True)
        self.samples = []
        self.load_profile()

    def load_profile(self):
        """Load existing voice samples"""
        if os.path.exists(os.path.join(self.profile_dir, "samples.txt")):
            with open(os.path.join(self.profile_dir, "samples.txt"), "r") as f:
                for line in f:
                    audio_file, text = line.strip().split("|")
                    if os.path.exists(audio_file):
                        self.samples.append((audio_file, text))

    def add_sample(self, audio_file, text):
        """Add new voice sample"""
        # Copy audio file to profile directory
        new_audio_path = os.path.join(self.profile_dir, f"sample_{len(self.samples)}.wav")
        AudioSegment.from_wav(audio_file).export(new_audio_path, format="wav")
        
        # Add to samples list
        self.samples.append((new_audio_path, text))
        
        # Save samples list
        with open(os.path.join(self.profile_dir, "samples.txt"), "w") as f:
            for audio, txt in self.samples:
                f.write(f"{audio}|{txt}\n")

    def get_combined_reference(self):
        """Combine all samples into a single reference"""
        if not self.samples:
            return None, ""
        
        combined_audio = AudioSegment.empty()
        combined_text = ""
        
        for audio_file, text in self.samples:
            audio_seg = AudioSegment.from_wav(audio_file)
            combined_audio += audio_seg + AudioSegment.silent(duration=500)  # Add 0.5s silence between samples
            combined_text += text + " "
        
        # Save combined audio to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        combined_audio.export(temp_file.name, format="wav")
        
        return temp_file.name, combined_text.strip()

class VoiceSamplesDataset(Dataset):
    def __init__(self, voice_profile):
        self.samples = voice_profile.samples
        self.mel_spec = MelSpec(
            n_fft=1024,
            hop_length=256, 
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24000,
            mel_spec_type="vocos"
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, text = self.samples[idx]
        audio, sr = torchaudio.load(audio_path)
        mel = self.mel_spec(audio)
        return {
            'text': text,
            'mel': mel,
            'audio': audio
        }


class OGF5TTSService:
    def __init__(self, model_dir, voice_profile, use_adapter=False, adapter_path=None, vocoder=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.voice_profile = VoiceProfile(voice_profile)
        self.use_adapter = use_adapter
        
        # Use provided vocoder or load a new one
        self.vocoder = vocoder if vocoder is not None else load_vocoder(vocoder_name="vocos", is_local=False)
        
        # Initialize model configuration
        if not use_adapter:
            # Original F5-TTS configuration
            model_cfg = dict(
                dim=1024,
                depth=22,
                heads=16,
                ff_mult=2,
                text_dim=512,
                conv_layers=4
            )

            mel_spec_kwargs = dict(
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                n_mel_channels=100,
                target_sample_rate=24000,
                mel_spec_type="vocos"
            )

            # Load vocabulary for text-based model
            vocab_char_map, vocab_size = load_vocab("vocab.txt")
            
            # Create and load original model
            self.model = CFM(
                transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
                mel_spec_kwargs=mel_spec_kwargs,
                vocab_char_map=vocab_char_map,
            ).to(self.device)
            
            checkpoint = torch.load(os.path.join(model_dir, "final_finetuned_model.pt"), map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        else:
            # Adapter-based configuration
            if adapter_path is None:
                raise ValueError("adapter_path must be provided when use_adapter is True")
                
            # Load adapter model
            from adapter.adapter import EnhancedEmbeddingAdapter
            self.model = EnhancedEmbeddingAdapter(
                llama_dim=3072,  # LLaMA hidden dimension
                tts_dim=1024,    # F5-TTS dimension
            ).to(self.device)
            
            checkpoint = torch.load(adapter_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get reference audio and text
        temp_ref_path, self.ref_text = self.voice_profile.get_combined_reference()
        
        if not temp_ref_path:
            raise ValueError("No voice samples found in profile")
            
        # Load and preprocess reference audio
        audio, sr = torchaudio.load(temp_ref_path)
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)
            
        # Convert to mono if needed
        if audio.dim() == 2 and audio.size(0) > 1:  # If stereo
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Store preprocessed audio tensor
        self.ref_audio = audio.to(self.device)
        
        # Clean up temporary file
        os.unlink(temp_ref_path)

    def synthesize(self, text=None, embeddings=None):
        """
        Synthesize speech from either text or embeddings
        """
        if embeddings is not None and self.use_adapter:
            return self.synthesize_from_embeddings(embeddings)
        elif text is not None and not self.use_adapter:
            return self.synthesize_from_text(text)
        else:
            raise ValueError(
                "Must provide embeddings when using adapter mode, "
                "or text when using original mode"
            )

    def synthesize_from_embeddings(self, embeddings):
        """Synthesize speech using adapter and embeddings"""
        if not self.use_adapter:
            raise ValueError("Cannot use embeddings with original F5-TTS mode")
            
        embeddings = embeddings.to(self.device)
        
        # Save reference audio to temporary file for infer_process
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            torchaudio.save(temp_file.name, self.ref_audio.cpu(), target_sample_rate)
            temp_path = temp_file.name
        
        try:
            audio_output, _, _ = infer_process(
                temp_path,
                self.ref_text,
                "",  # Empty gen_text since using embeddings
                self.model,
                self.vocoder,
                conditioning_input=embeddings,
                mel_spec_type="vocos",
                speed=1.0,
                nfe_step=32,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
                device=self.device
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return audio_output

    def synthesize_from_text(self, text):
        """Synthesize speech using original F5-TTS with text input"""
        if self.use_adapter:
            raise ValueError("Cannot use text input with adapter mode")
            
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            torchaudio.save(temp_file.name, self.ref_audio.cpu(), target_sample_rate)
            temp_path = temp_file.name
            
        try:
            audio_output, _, _ = infer_process(
                temp_path,
                self.ref_text,
                text,
                self.model,
                self.vocoder,
                mel_spec_type="vocos",
                speed=1.0,
                nfe_step=32,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
                device=self.device
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
        return audio_output

    def save_audio(self, audio, filename, sample_rate=24000):
        """Save audio to file"""
        if isinstance(audio, tuple):
            audio, _ = audio
        
        if isinstance(audio, list):
            audio = audio[0]
        
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()
        
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        audio = audio.astype(np.float32)
        
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        sf.write(filename, audio, sample_rate)

def main():
    print("F5-TTS Voice Cloning System (Continuous Learning)")
    print("---------------------------------------------")
    
    profile_name = "Dobby"
    profile = VoiceProfile(profile_name)
    
    # Get combined reference
    ref_audio_path, ref_text = profile.get_combined_reference()
        
    # Preprocess reference audio and text
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
    os.unlink(ref_audio_path)
                
    text = "testing out voice synthesis"
    
    print("\nGenerating speech...")
    # Generate speech using F5-TTS utilities
    audio, sample_rate, _ = infer_process(
        ref_audio,
        ref_text,
        text,
        model,
        vocoder,
        mel_spec_type="vocos",
        speed=1.0,
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0
    )
    
    # Play synthesized audio
    sd.play(audio, samplerate=sample_rate)
    sd.wait()
                    

if __name__ == "__main__":
    main()