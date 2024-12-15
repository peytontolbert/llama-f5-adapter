import torch
import torch.nn.functional as F

def analyze_mel(mel):
    """Analyze mel spectrogram properties"""
    print(f"\nMel Spectrogram Analysis:")
    print(f"Shape: {mel.shape}")
    print(f"Range: {mel.min():.3f} to {mel.max():.3f}")
    print(f"Mean: {mel.mean():.3f}")
    print(f"Std: {mel.std():.3f}")
    
    # Check for silence/dead channels
    channel_means = mel.mean(dim=-1)  # Average across time
    dead_channels = (channel_means.abs() < 1e-6).sum()
    print(f"Dead channels: {dead_channels}")
    
    # Check temporal structure
    temporal_diff = torch.diff(mel, dim=-1).abs().mean()
    print(f"Average temporal difference: {temporal_diff:.3f}")
    
    # Check frequency structure
    freq_diff = torch.diff(mel, dim=1).abs().mean()
    print(f"Average frequency difference: {freq_diff:.3f}")
    
    # Check for NaN/Inf values
    has_nan = torch.isnan(mel).any()
    has_inf = torch.isinf(mel).any()
    if has_nan or has_inf:
        print("Warning: NaN or Inf values detected!")
    
    return {
        'range': (mel.min().item(), mel.max().item()),
        'mean': mel.mean().item(),
        'std': mel.std().item(),
        'dead_channels': dead_channels.item(),
        'temporal_diff': temporal_diff.item(),
        'freq_diff': freq_diff.item(),
        'has_nan': has_nan.item(),
        'has_inf': has_inf.item()
    }

def check_mel_compatibility(mel, target_range=(-12, 2)):
    """Check if mel spectrogram matches F5-TTS expectations"""
    stats = analyze_mel(mel)
    
    issues = []
    
    # Check value range
    if stats['range'][0] < target_range[0] or stats['range'][1] > target_range[1]:
        issues.append(f"Mel range {stats['range']} outside target range {target_range}")
    
    # Check for dead channels
    if stats['dead_channels'] > 0:
        issues.append(f"Found {stats['dead_channels']} dead channels")
        
    # Check temporal structure
    if stats['temporal_diff'] < 0.1:
        issues.append("Low temporal variation - possible static output")
        
    # Check frequency structure  
    if stats['freq_diff'] < 0.1:
        issues.append("Low frequency variation - possible tonal issues")
        
    # Check for NaN/Inf
    if stats['has_nan'] or stats['has_inf']:
        issues.append("Contains NaN or Inf values")
        
    return issues

def plot_mel(mel, title="Mel Spectrogram"):
    """Plot mel spectrogram for visualization"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        plt.imshow(mel.squeeze().cpu().numpy(), 
                  aspect='auto', 
                  origin='lower',
                  interpolation='nearest')
        plt.colorbar(label='Amplitude')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Mel Channel')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not installed - skipping plot") 

def compute_mel_quality_metrics(pred_mel, target_mel):
    """Compute comprehensive mel quality metrics"""
    metrics = {}
    
    # Spectral convergence
    metrics['spectral_convergence'] = torch.norm(
        target_mel - pred_mel, p='fro') / torch.norm(target_mel, p='fro')
    
    # Log-magnitude error
    metrics['log_magnitude_error'] = F.l1_loss(
        torch.log1p(torch.abs(pred_mel)),
        torch.log1p(torch.abs(target_mel))
    )
    
    # Envelope similarity
    pred_env = torch.norm(pred_mel, dim=1)
    target_env = torch.norm(target_mel, dim=1)
    metrics['envelope_similarity'] = F.cosine_similarity(
        pred_env, target_env, dim=-1).mean()
    
    # Frequency-bin correlation
    for bin_idx in range(pred_mel.size(1)):
        metrics[f'freq_corr_{bin_idx}'] = F.cosine_similarity(
            pred_mel[:, bin_idx], 
            target_mel[:, bin_idx], 
            dim=-1
        ).mean()
    
    return metrics