import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from f5.f5tts import F5TTSService
from adapter.adapter import EnhancedEmbeddingAdapter
from f5.utils_infer import infer_process, load_vocoder
from f5.utils import lens_to_mask
import os
import tempfile

def load_adapter(checkpoint_path, llama_dim=3072, tts_dim=1024, device="cuda"):
    """Load trained adapter model"""
    adapter = EnhancedEmbeddingAdapter(
        llama_dim=llama_dim,
        tts_dim=tts_dim
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    adapter.load_state_dict(checkpoint['model_state_dict'])
    adapter.eval()
    
    print(f"Loaded adapter from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    return adapter

def generate_speech(text, llama_model, tokenizer, adapter, f5_service, device):
    print(f"\nGenerating speech for: {text}")
    
    # Generate creative text from input prompt
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        sequences = llama_model.generate(
            inputs.input_ids,
            min_length=32,
            max_length=50,
            temperature=0.9,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Show generated text
        generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
        print(f"\nGenerated text ({len(sequences[0])} tokens):")
        print(generated_text)
        
        # Get embeddings from generated text
        outputs = llama_model(sequences, output_hidden_states=True)
        llama_embeddings = outputs.hidden_states[-1]
    
    print(f"\nLLaMA embeddings shape: {llama_embeddings.shape}")
    
    # Generate mel spectrograms using adapter
    with torch.no_grad():
        timesteps = torch.zeros(llama_embeddings.shape[0], device=device)
        mel_output = adapter(
            llama_embeddings,
            timesteps=timesteps
        )
    
    print(f"Generated mel spectrogram shape: {mel_output.shape}")
    print(f"Mel range: min={mel_output.min().item():.3f}, max={mel_output.max().item():.3f}")
    
    # Convert mel spectrograms to audio using vocoder
    with torch.no_grad():
        audio = f5_service.vocoder.decode(mel_output)
        audio = audio.squeeze().cpu()
    
    print(f"Generated audio shape: {audio.shape}")
    return audio.numpy()

def blend_chunks(chunks, overlap):
    """Blend overlapping mel spectrogram chunks smoothly"""
    if len(chunks) == 1:
        return chunks[0]
    
    # Calculate total length
    total_len = sum(chunk.shape[1] for chunk in chunks) - overlap * (len(chunks) - 1)
    
    # Initialize output tensor
    output = torch.zeros(
        (chunks[0].shape[0], total_len, chunks[0].shape[2]),
        device=chunks[0].device
    )
    
    # Create blending weights for overlap regions
    overlap_weights = torch.linspace(0, 1, overlap, device=chunks[0].device)
    
    current_pos = 0
    for i, chunk in enumerate(chunks):
        if i == 0:
            # First chunk
            output[:, :chunk.shape[1] - overlap] = chunk[:, :-overlap]
            output[:, chunk.shape[1] - overlap:chunk.shape[1]] = (
                chunk[:, -overlap:] * (1 - overlap_weights).unsqueeze(0).unsqueeze(-1) +
                chunks[i + 1][:, :overlap] * overlap_weights.unsqueeze(0).unsqueeze(-1)
            )
        elif i == len(chunks) - 1:
            # Last chunk
            output[:, current_pos + overlap:] = chunk[:, overlap:]
        else:
            # Middle chunks
            output[:, current_pos + overlap:current_pos + chunk.shape[1] - overlap] = chunk[:, overlap:-overlap]
            output[:, current_pos + chunk.shape[1] - overlap:current_pos + chunk.shape[1]] = (
                chunk[:, -overlap:] * (1 - overlap_weights).unsqueeze(0).unsqueeze(-1) +
                chunks[i + 1][:, :overlap] * overlap_weights.unsqueeze(0).unsqueeze(-1)
            )
        current_pos += chunk.shape[1] - overlap
    
    return output

def convert_checkpoint(old_checkpoint_path, new_checkpoint_path):
    checkpoint = torch.load(old_checkpoint_path)
    new_state_dict = {}
    
    # Copy compatible layers
    for k, v in checkpoint['model_state_dict'].items():
        if not any(x in k for x in ['pos_embed', 'time_embed', 'final_norm']):
            new_state_dict[k] = v
            
    # Add new layers
    adapter = EnhancedEmbeddingAdapter()
    missing_keys = adapter.state_dict().keys() - new_state_dict.keys()
    for k in missing_keys:
        new_state_dict[k] = adapter.state_dict()[k]
        
    checkpoint['model_state_dict'] = new_state_dict
    torch.save(checkpoint, new_checkpoint_path)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Initialize models and tokenizer
    print("Loading models...")
    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3b").to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    # Load adapter with same config as training
    print("\nLoading adapter...")
    adapter = EnhancedEmbeddingAdapter(
        llama_dim=3072,
        tts_dim=1024,
        depth=8,
        heads=8,
        dim_head=64,
        ff_mult=4
    ).to(device)
    
    # Load trained adapter weights
    checkpoint = torch.load("checkpoints/adapter_epoch_3.pt", map_location=device)
    adapter.load_state_dict(checkpoint['model_state_dict'])
    adapter.eval()
    
    print("\nLoading F5-TTS vocoder...")
    # Only load vocoder from F5TTS
    f5_service = F5TTSService(
        model_dir="weights",
        voice_profile="Bob",
        load_base_model=False  # Don't load the base model
    )
    
    # Generate speech
    audio = generate_speech(
        text="Tell me an exciting story about dragons",
        llama_model=llama_model,
        tokenizer=tokenizer,
        adapter=adapter,
        f5_service=f5_service,
        device=device
    )
    
    # Save audio
    torchaudio.save("output.wav", torch.tensor(audio).unsqueeze(0), 24000)
    print(f"\nGenerated speech saved to output.wav")

if __name__ == "__main__":
    main() 