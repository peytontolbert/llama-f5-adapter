import torch
import torch.nn.functional as F
from adapter.new_adapter import CSMAdapter
from f5.utils_infer import load_vocoder
import torchaudio
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm

class CSMEvaluator:
    def __init__(self, checkpoint_path, device='cuda:0'):
        self.device = device
        
        # Load adapter
        self.adapter = CSMAdapter(
            llama_dim=3072,
            tts_dim=1024,
            depth=8,
            heads=8,
            dim_head=64,
            n_mel_channels=100
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.adapter.load_state_dict(checkpoint['model_state_dict'])
        self.adapter.eval()
        
        # Load vocoder
        self.vocoder = load_vocoder("vocos")
        
    def generate_audio(self, llama_embeddings, save_path=None):
        """
        Generate audio from LLaMA embeddings
        """
        self.adapter.eval()
        audio = None
        
        with torch.no_grad():
            # Add batch dimension
            embeddings = llama_embeddings.unsqueeze(0).to(self.device)
            
            # Generate mel spectrogram
            mel = self.adapter(
                embeddings,
                timesteps=torch.zeros(1, device=self.device),
                mask=torch.ones(1, embeddings.size(1), dtype=torch.bool, device=self.device)
            )
            
            print(f"Initial mel shape: {mel.shape}")
            
            # Handle potential 4D output and ensure contiguous memory
            if len(mel.shape) == 4:
                mel = mel.contiguous().view(mel.size(0), mel.size(1), -1)
            
            # Ensure mel has shape [B x n_mels x T]
            if mel.size(1) != 100:
                mel = mel.transpose(1, 2).contiguous()
            
            # Normalize mel spectrogram
            mel = 2 * (mel - mel.min()) / (mel.max() - mel.min() + 1e-8) - 1
            
            print("\nVocoder info:")
            print(f"Vocoder type: {type(self.vocoder)}")
            print(f"Feature extractor type: {type(self.vocoder.feature_extractor)}")
            
            try:
                # Try using mel spectrogram directly
                print(f"\nProcessed mel shape before vocoder: {mel.shape}")
                
                # Keep in [B x n_mels x T] format for vocoder
                print(f"Mel shape for vocoder: {mel.shape}")
                
                # Generate audio using vocoder's decode method
                audio = self.vocoder.decode(mel)
                
            except RuntimeError as e:
                print(f"Direct vocoder error: {str(e)}")
                try:
                    # Try alternative approach
                    print("Trying alternative approach...")
                    
                    # Ensure mel has correct number of channels
                    if mel.size(1) != 100:
                        mel = F.interpolate(
                            mel,
                            size=(100, mel.size(-1)),
                            mode='bilinear',
                            align_corners=False
                        )
                    print(f"Reshaped mel shape: {mel.shape}")
                    
                    # Generate audio
                    audio = self.vocoder.decode(mel)
                    
                except Exception as e2:
                    print(f"Alternative approach failed: {str(e2)}")
                    try:
                        # Try one last approach
                        print("Trying final approach...")
                        
                        # Use raw mel input with channel adjustment
                        mel = mel.squeeze(0)  # [n_mels x T]
                        if mel.size(0) != 100:
                            mel = F.interpolate(
                                mel.unsqueeze(0),  # [1 x n_mels x T]
                                size=(100, mel.size(-1)),
                                mode='bilinear',
                                align_corners=False
                            )
                        mel = mel.squeeze(0)  # [n_mels x T]
                        mel = mel.unsqueeze(0)  # [1 x n_mels x T]
                        
                        print(f"Final mel shape: {mel.shape}")
                        
                        # Generate audio
                        audio = self.vocoder.decode(mel)
                        
                    except Exception as e3:
                        print(f"All attempts failed: {str(e3)}")
                        # Return silence with correct length
                        audio = torch.zeros(1, int(mel.size(-1) * 256))
                
                # Scale audio back to original length if needed
                target_length = int(llama_embeddings.size(0) * 256)
                if audio is not None and audio.size(-1) != target_length:
                    audio = F.interpolate(
                        audio.unsqueeze(1),
                        size=target_length,
                        mode='linear',
                        align_corners=False
                    ).squeeze(1)
                
                if save_path and audio is not None:
                    torchaudio.save(save_path, audio.cpu(), 24000)

        return audio.squeeze().cpu(), 24000 if audio is not None else (None, None)
        
    def evaluate_dataset(self, dataset_path, output_dir="eval_outputs"):
        """
        Evaluate adapter on a test dataset
        """
        # Load dataset
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
            if isinstance(dataset, dict):
                dataset = dataset['data']
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        metrics = {
            'mel_l1': [],
            'mel_l2': [],
            'duration_error': []
        }
        
        try:
            print("Evaluating samples...")
            for i, sample in enumerate(tqdm(dataset)):
                try:
                    # Generate mel spectrogram
                    with torch.no_grad():
                        seq_len = sample['embeddings'].size(0)
                        
                        pred_mel, pred_durations = self.adapter(
                            sample['embeddings'].unsqueeze(0).to(self.device),
                            timesteps=torch.zeros(1, device=self.device),
                            mask=torch.ones(1, seq_len, dtype=torch.bool, device=self.device),
                            return_durations=True
                        )
                        
                        # Handle potential 4D output
                        if len(pred_mel.shape) == 4:
                            pred_mel = pred_mel.squeeze(2)
                        
                        # Interpolate to match target length
                        pred_mel = F.interpolate(
                            pred_mel,
                            size=sample['mel_spec'].size(-1),
                            mode='linear',
                            align_corners=False
                        )
                    
                    # Print shapes for debugging
                    print(f"\nSample {i}:")
                    print(f"pred_mel shape: {pred_mel.shape}")
                    print(f"target_mel shape: {sample['mel_spec'].shape}")
                    
                    # Compute mel metrics
                    metrics['mel_l1'].append(F.l1_loss(pred_mel, sample['mel_spec'].unsqueeze(0).to(self.device)).item())
                    metrics['mel_l2'].append(F.mse_loss(pred_mel, sample['mel_spec'].unsqueeze(0).to(self.device)).item())
                    
                    # Compute duration error if available
                    if 'alignment' in sample and 'token_durations' in sample['alignment']:
                        target_durations = sample['alignment']['token_durations'].to(self.device)
                        min_len = min(pred_durations.size(1), len(target_durations))
                        pred_dur = pred_durations.squeeze(0)[:min_len]
                        target_dur = target_durations[:min_len]
                        metrics['duration_error'].append(F.l1_loss(pred_dur, target_dur).item())
                    
                    # Generate and save audio
                    audio, sr = self.generate_audio(sample['embeddings'])
                    torchaudio.save(
                        output_dir / f"sample_{i}.wav",
                        audio.unsqueeze(0),
                        sr
                    )
                    
                    # Save mel spectrograms for comparison
                    np.save(output_dir / f"pred_mel_{i}.npy", pred_mel.cpu().numpy())
                    np.save(output_dir / f"target_mel_{i}.npy", sample['mel_spec'].cpu().numpy())
                    
                    # Save durations for comparison
                    if 'alignment' in sample and 'token_durations' in sample['alignment']:
                        np.save(output_dir / f"pred_durations_{i}.npy", pred_durations.cpu().numpy())
                        np.save(output_dir / f"target_durations_{i}.npy", target_durations.cpu().numpy())
                        
                except Exception as e:
                    print(f"\nError processing sample {i}: {str(e)}")
                    print(f"Input embeddings shape: {sample['embeddings'].shape}")
                    continue
            
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user")
        except Exception as e:
            print(f"\nEvaluation failed: {str(e)}")
        finally:
            # Print metrics for completed samples
            print("\nEvaluation Results (partial):")
            for metric, values in metrics.items():
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    print(f"{metric}: {mean:.4f} ± {std:.4f}")
            
            return metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='best_csm_adapter.pt',
                      help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='generated_dataset.pkl',
                      help='Path to evaluation dataset')
    parser.add_argument('--output_dir', type=str, default='eval_outputs',
                      help='Directory to save evaluation outputs')
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Device to run evaluation on')
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}")
    evaluator = CSMEvaluator(args.checkpoint, device=args.device)
    
    print(f"Evaluating on dataset: {args.dataset}")
    evaluator.evaluate_dataset(args.dataset, args.output_dir)

if __name__ == '__main__':
    main() 