import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from f5.f5tts import F5TTSService, model as f5_model
from adapter.adapter import EnhancedEmbeddingAdapter
from f5.utils import list_str_to_idx

def test_dimensions():
    print("\nTesting dimensions throughout the pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize models
    print("Loading LLaMA model...")
    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3b").to(device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading F5-TTS model...")
    f5_service = F5TTSService(model_dir="f5/weights", voice_profile="Dobby")
    
    print("Initializing adapter...")
    adapter = EnhancedEmbeddingAdapter(
        llama_dim=3072,
        tts_dim=1024,
        depth=8,
        heads=8,
        dim_head=64,
        ff_mult=4,
        n_mel_channels=f5_model.num_channels
    ).to(device)
    
    # Test text generation and embedding extraction
    print("\nTesting LLaMA generation...")
    test_prompt = "Tell me a story about"
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = llama_model.generate(
            **inputs,
            min_length=32,  # Ensure minimum length for vocoder
            max_length=100,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        
        # Get LLaMA embeddings for each token
        llama_embeddings = outputs.hidden_states[-1][-1]  # [1, seq_len, 3072]
        print(f"LLaMA embeddings shape: {llama_embeddings.shape}")
        
        # Create length tensor for masking
        seq_lengths = torch.tensor([llama_embeddings.shape[1]], device=device)
        
        # Test adapter with proper masking
        print("\nTesting adapter...")
        timesteps = torch.zeros((llama_embeddings.shape[0],), device=device)
        adapter_output = adapter(llama_embeddings, timesteps)
        
        # Repeat the adapter output to a reasonable length for vocoder
        min_length = 32  # Minimum length needed for vocoder's convolutions
        if adapter_output.shape[1] < min_length:
            adapter_output = adapter_output.repeat(1, min_length, 1)
            
        print(f"Adapter output shape: {adapter_output.shape}")
        
        # Generate audio using F5TTSService
        print("\nGenerating audio...")
        audio_output = f5_service.synthesize_from_embeddings(
            adapter_output,
            # No text conditioning - working directly with embeddings
        )
        print(f"Final audio shape: {audio_output[0].shape}")

if __name__ == "__main__":
    test_dimensions() 