import torch
import torch.nn as nn
import torch.nn.functional as F
from f5.modules import (
    DiTBlock,
    TimestepEmbedding,
    ConvPositionEmbedding,
    AdaLayerNormZero,
    GRN,
    RelativePositionalEmbedding,
    AttnProcessor,
    Attention
)
import math
from f5.utils import lens_to_mask
from torch.utils.data import Dataset

def lens_to_mask(lens, length=None):
    """Convert lengths to mask.
    
    Args:
        lens: [batch_size] tensor of sequence lengths
        length: optional max length for the mask
        
    Returns:
        [batch_size, max_length] boolean mask
    """
    if length is None:
        length = lens.max()
    mask = torch.arange(length, device=lens.device)[None, :] < lens[:, None]
    return mask

class LengthRegulator(nn.Module):
    def forward(self, x, durations, seq_lengths=None):
        """
        Args:
            x: Input tensor [B, T, C]
            durations: Duration predictions [B, T]
            seq_lengths: Valid sequence lengths [B]
        """
        if seq_lengths is not None:
            # Mask durations beyond sequence length
            mask = torch.arange(durations.size(1), device=durations.device)[None, :] < seq_lengths[:, None]
            durations = durations * mask.float()
        
        # Calculate output lengths
        output_lengths = torch.round(durations.sum(dim=1)).long()
        max_len = output_lengths.max()
        
        # Regulate lengths
        output = torch.zeros(
            x.shape[0], max_len, x.shape[2],
            device=x.device, dtype=x.dtype
        )
        
        for b in range(x.shape[0]):
            cur_pos = 0
            for t in range(seq_lengths[b] if seq_lengths is not None else x.shape[1]):
                dur = int(durations[b, t].item())
                if dur > 0:
                    output[b, cur_pos:cur_pos + dur] = x[b, t]
                    cur_pos += dur
        
        return output

class EnhancedEmbeddingAdapter(nn.Module):
    """
    Enhanced adapter that converts LLaMA embeddings to F5-TTS compatible embeddings
    using DiT-style blocks and CFM-compatible outputs
    """
    def __init__(
        self,
        llama_dim=3072,
        tts_dim=1024,
        depth=8,
        heads=8,
        dim_head=64,
        ff_mult=4,
        n_mel_channels=100,
        min_seq_len=32,
        dropout=0.1
    ):
        super().__init__()
        
        print(f"\nInitializing adapter with:")
        print(f"llama_dim: {llama_dim}")
        print(f"tts_dim: {tts_dim}")
        print(f"n_mel_channels: {n_mel_channels}")
        
        # Pre-norm layers
        self.input_norm = nn.LayerNorm(llama_dim)
        self.content_norm = nn.LayerNorm(tts_dim)
        self.prosody_norm = nn.LayerNorm(tts_dim)
        
        # Improved input projection with residual
        self.input_proj = nn.Sequential(
            nn.Linear(llama_dim, tts_dim * 2),
            nn.GELU(),
            nn.Linear(tts_dim * 2, tts_dim),
            nn.LayerNorm(tts_dim)
        )
        
        # Text encoder blocks (like F5-TTS)
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                ConvNeXtV2Block(tts_dim, tts_dim*4),
                AdaLayerNormZero(tts_dim),
                DiTBlock(tts_dim, heads=8, use_relative_pos=True)
            ) for _ in range(depth//2)
        ])
        
        # Add length regulator
        self.length_regulator = LengthRegulator()
        
        # Initialize duration predictor
        self.duration_predictor = DurationPredictor(
            hidden_dim=tts_dim,
            input_dim=llama_dim
        )
        
        # Then use it in variance adaptor
        self.variance_adaptor = nn.ModuleList([
            ConvNeXtV2Block(tts_dim, tts_dim*4),
            AdaLayerNormZero(tts_dim),
            self.duration_predictor
        ])
        
        # Decoder blocks (F5-TTS style)
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                ConvNeXtV2Block(tts_dim, tts_dim*4),
                AdaLayerNormZero(tts_dim),
                DiTBlock(tts_dim, heads=8)
            ) for _ in range(depth//2)
        ])
        
        # Output layers
        self.final_norm = nn.LayerNorm(tts_dim)
        self.to_mel = nn.Linear(tts_dim, n_mel_channels)
        
        # Add prosody encoder
        self.prosody_encoder = ProsodyEncoder(tts_dim)
        
        # Add time embedding layer
        self.time_embed = TimestepEmbedding(
            dim=tts_dim
        )
        
        # Add mel range normalization
        self.mel_range_norm = MelRangeNormalization(n_mel_channels)
        
        # Add dropout and layer norm
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(tts_dim)
        
        # Add F5-TTS style position encoding
        self.rel_pos = RelativePositionalEmbedding(
            dim=tts_dim,
            num_heads=heads,
            max_positions=2048
        )
        self.conv_pos = ConvPositionEmbedding(tts_dim)
        
        # Add temporal attention
        self.temporal_attn = nn.MultiheadAttention(
            tts_dim, 
            heads, 
            dropout=0.1,
            batch_first=True
        )
        
        # Add temporal variation modules
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(tts_dim, tts_dim, 3, padding=1, groups=tts_dim),
            nn.ReLU(),
            nn.Conv1d(tts_dim, tts_dim, 3, padding=1, groups=tts_dim)
        )
        
        # Reference encoder for style matching
        self.ref_style_encoder = nn.Sequential(
            nn.Conv1d(n_mel_channels, tts_dim, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(tts_dim),
            nn.Conv1d(tts_dim, tts_dim, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(tts_dim)
        )
        
        # Rest of your initialization code...
        self.target_len = 32  # Add fixed target length

    def forward(self, llama_embeddings, timesteps, mask=None, return_durations=False, ref_audio=None, ref_text=None):
        # Get time embeddings
        time_emb = self.time_embed(timesteps)
        
        # Ensure input has target length of 32
        if llama_embeddings.size(1) != self.target_len:
            llama_embeddings = F.interpolate(
                llama_embeddings.transpose(1, 2),
                size=self.target_len,
                mode='nearest'
            ).transpose(1, 2)
        
        # Encode reference style if available
        ref_style = None
        if ref_audio is not None:
            ref_style = self.ref_style_encoder(ref_audio.transpose(1, 2))
            ref_style = ref_style.transpose(1, 2)
            
            # Ensure reference matches target length
            if ref_style.size(1) != self.target_len:
                ref_style = F.interpolate(
                    ref_style.transpose(1, 2),
                    size=self.target_len,
                    mode='nearest'
                ).transpose(1, 2)
        
        # Text encoding
        x = self.input_proj(llama_embeddings)
        
        # Ensure mask matches target length
        if mask is not None:
            if mask.size(1) != self.target_len:
                mask = F.interpolate(
                    mask.float().unsqueeze(1),
                    size=self.target_len,
                    mode='nearest'
                ).squeeze(1).bool()
        
        # Process through encoder blocks
        for block in self.encoder_blocks:
            # Apply temporal convolution with length preservation
            x_temp = x.transpose(1, 2)  # [B, C, T]
            x_temp = self.temporal_conv(x_temp)  # [B, C, T] - Length preserved due to padding
            x_temp = x_temp.transpose(1, 2)  # [B, T, C]
            
            # Create attention masks
            if mask is not None:
                if len(mask.shape) == 3:
                    mask = mask.squeeze(1)
                attn_mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
                attn_mask = attn_mask.repeat(self.temporal_attn.num_heads, 1, 1)
            else:
                attn_mask = None
            
            # Apply temporal attention
            temporal_out = self.temporal_attn(
                x,  # [batch, 32, dim]
                x_temp,  # [batch, 32, dim] - Now guaranteed same length as x
                x_temp,  # [batch, 32, dim]
                attn_mask=attn_mask,
                need_weights=False
            )[0]
            
            # Controlled residual connection
            x = x + 0.1 * temporal_out
            
            # Regular block processing
            conv_out = block[0](x)
            norm_out, _, _ = block[1](conv_out, time_emb)
            x = block[2](norm_out, time_emb, mask=mask)
        
        # Get durations with proper masking
        durations = self.duration_predictor(x, ref_audio, ref_text, mask=mask)
        
        # Apply mask to durations if provided
        if mask is not None:
            durations = durations * mask.float()
        
        # Length regulation with safety checks
        try:
            # Get valid sequence lengths from mask
            if mask is not None:
                seq_lengths = mask.sum(dim=1)
            else:
                seq_lengths = torch.full((x.shape[0],), x.shape[1], device=x.device)
            
            # Apply length regulation
            x = self.length_regulator(x, durations, seq_lengths)
        except Exception as e:
            print(f"Length regulation error: {str(e)}")
            # Fallback to simple upsampling
            target_len = int(durations.sum(dim=1).max().item())
            x = F.interpolate(
                x.transpose(1, 2),
                size=target_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        # Process through decoder blocks
        for block in self.decoder_blocks:
            conv_out = block[0](x)
            norm_out, _, _ = block[1](conv_out, time_emb)
            x = block[2](norm_out, time_emb)
        
        # Generate mel with gradient clipping
        mel = self.to_mel(x)
        mel = torch.clamp(mel, min=-100, max=100)  # Prevent extreme values
        mel = mel.transpose(1, 2)
        
        if return_durations:
            return mel, durations
        return mel

    def adjust_model(self, convergence_metrics):
        """Adjust model based on convergence analysis"""
        if convergence_metrics['mel_range'] > 1.0:
            # Adjust mel range normalization
            self.mel_range_norm.scale.data *= 0.95
        
        if convergence_metrics['duration_mean'] > 0.5:
            # Adjust duration predictor bias
            self.duration_predictor.proj.bias.data *= 0.98

class MelRangeNormalization(nn.Module):
    def __init__(self, n_mel_channels):
        super().__init__()
        self.register_buffer('target_min', torch.tensor(-12.0))
        self.register_buffer('target_max', torch.tensor(2.0))
        
        # Learnable scaling parameters per channel
        self.scale = nn.Parameter(torch.ones(1, n_mel_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, n_mel_channels, 1))
        self.range_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, mel):
        # Dynamic range adjustment
        mel = mel * self.scale + self.bias
        
        # Scale to target range
        mel = torch.tanh(mel * self.range_scale)
        mel = mel * (self.target_max - self.target_min)/2 + (self.target_max + self.target_min)/2
        
        return mel

class DurationPredictor(nn.Module):
    def __init__(self, hidden_dim, input_dim=3072):
        super().__init__()
        self.target_len = 32  # Add fixed target length
        
        # Bi-directional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim//2,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True
        )
        
        # Reference encoder with attention
        self.ref_encoder = nn.Sequential(
            nn.Conv1d(100, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Multi-head attention for aligning with reference
        self.ref_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Duration predictor network
        self.duration_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # Variance predictor for temporal variation
        self.variance_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
    def forward(self, x, ref_audio=None, ref_text=None, mask=None):
        # Ensure input has target length
        if x.size(1) != self.target_len:
            x = F.interpolate(
                x.transpose(1, 2),
                size=self.target_len,
                mode='nearest'
            ).transpose(1, 2)
        
        # Process through LSTM
        x_lstm, _ = self.lstm(x)
        
        # Process reference if available
        if ref_audio is not None:
            # Encode reference mel
            ref_features = self.ref_encoder(ref_audio.transpose(1, 2))
            ref_features = ref_features.transpose(1, 2)
            
            # Ensure reference has target length
            if ref_features.size(1) != self.target_len:
                ref_features = F.interpolate(
                    ref_features.transpose(1, 2),
                    size=self.target_len,
                    mode='nearest'
                ).transpose(1, 2)
            
            # Create attention mask
            if mask is not None:
                if len(mask.shape) == 3:
                    mask = mask.squeeze(1)
                
                # Create fixed size attention mask
                attn_mask = torch.zeros(
                    (x.shape[0], self.target_len, self.target_len),
                    device=x.device,
                    dtype=torch.bool
                )
                
                # Fill valid positions
                for i in range(x.shape[0]):
                    valid_len = min(mask[i].sum().item(), self.target_len)
                    attn_mask[i, :valid_len, :valid_len] = True
            else:
                attn_mask = None
            
            # Attend to reference
            attn_output = self.ref_attention(
                x_lstm,
                ref_features,
                ref_features,
                attn_mask=attn_mask,
                need_weights=False
            )[0]
            
            x_combined = torch.cat([x_lstm, attn_output], dim=-1)
        else:
            x_combined = torch.cat([x_lstm, torch.zeros_like(x_lstm)], dim=-1)
        
        # Predict durations
        durations = self.duration_net(x_combined)
        
        # Predict variance
        variance = self.variance_net(x_lstm)
        variance = torch.clamp(variance, min=-0.5, max=0.5)
        
        # Apply variance to durations
        durations = durations * (1.0 + 0.05 * variance)
        
        # Ensure positive durations
        durations = F.softplus(durations) + 1e-6
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has target length
            if mask.size(1) != self.target_len:
                mask = F.interpolate(
                    mask.float().unsqueeze(1),
                    size=self.target_len,
                    mode='nearest'
                ).squeeze(1).bool()
            durations = durations * mask.float()
        
        return durations.squeeze(-1)

    def infer_durations(self, x, ref_audio=None, ref_text=None, mask=None, speed=1.0):
        """Special inference mode with speed control"""
        durations = self.forward(x, ref_audio, ref_text, mask)
        
        # Apply speed factor
        durations = durations / speed
        
        # Quantize to integer number of frames
        durations = torch.round(durations)
        
        # Ensure minimum duration
        durations = torch.clamp(durations, min=1)
        
        return durations

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale

class MelDiscriminator(nn.Module):
    def __init__(self, n_mel_channels):
        super().__init__()
        
        # Multi-scale discriminators
        self.discriminators = nn.ModuleList([
            self._make_discriminator(n_mel_channels, scale) 
            for scale in [1, 2, 4]  # Different scales for analysis
        ])
        
    def _make_discriminator(self, channels, scale):
        return nn.Sequential(
            nn.Conv2d(1, 32 * scale, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32 * scale, 64 * scale, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64 * scale, 128 * scale, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128 * scale, 1, kernel_size=(3, 3), padding=1)
        )
    
    def forward(self, mel):
        # Add channel dimension
        mel = mel.unsqueeze(1)  # [B, 1, M, T]
        
        # Get discriminator outputs at different scales
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(mel))
            
        return outputs

class ProsodyEncoder(nn.Module):
    def __init__(self, tts_dim):
        super().__init__()
        
        self.energy_encoder = nn.Sequential(
            nn.Conv1d(1, tts_dim//4, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(tts_dim//4, tts_dim//2, 3, padding=1),
            nn.ReLU()
        )
        
        self.pitch_encoder = nn.Sequential(
            nn.Conv1d(1, tts_dim//4, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(tts_dim//4, tts_dim//2, 3, padding=1),
            nn.ReLU()
        )
        
        self.combine = nn.Linear(tts_dim, tts_dim)
        
    def forward(self, mel):
        # Extract energy and pitch features
        energy = torch.norm(mel, dim=1, keepdim=True)
        energy_features = self.energy_encoder(energy)
        
        # Approximate pitch using autocorrelation
        pitch = self.compute_pitch(mel)
        pitch_features = self.pitch_encoder(pitch)
        
        # Combine features
        prosody = torch.cat([energy_features, pitch_features], dim=1)
        return self.combine(prosody.transpose(1, 2))

class VarianceAdaptor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.duration_predictor = DurationPredictor(dim)
        self.pitch_predictor = PitchPredictor(dim) 
        self.energy_predictor = EnergyPredictor(dim)
        self.length_regulator = LengthRegulator()

class F5AttentionBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            use_relative_pos=True
        )
        self.norm = AdaLayerNormZero(dim)

class MelDecoder(nn.Module):
    def __init__(self, dim, n_mel_channels):
        super().__init__()
        self.prenet = nn.Sequential(
            ConvNeXtV2Block(dim, dim*4),
            ConvNeXtV2Block(dim, dim*4),
            ConvNeXtV2Block(dim, dim*4)
        )
        self.postnet = nn.Conv1d(dim, n_mel_channels, 1)

class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        # Increase receptive field
        self.dwconv = nn.Sequential(
            nn.Conv1d(dim, dim, 5, padding=2, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, ff_dim)
        self.act = nn.GELU()
        self.grn = GRN(ff_dim)
        self.pwconv2 = nn.Linear(ff_dim, dim)
        
    def forward(self, x):
        residual = x
        
        # Add temporal attention
        if x.size(1) >= 4:
            x = x.transpose(1, 2)
            x = self.dwconv(x)
            x = x.transpose(1, 2)
            
            # Add temporal mixing
            x = x + 0.1 * torch.roll(x, shifts=1, dims=1)
        
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x

class PitchPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
        
    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.net(x)
        return x.transpose(1, 2)  # [B, T, 1]

class EnergyPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
        
    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.net(x)
        return x.transpose(1, 2)  # [B, T, 1]

class AdapterDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'embeddings': sample['embeddings'],
            'mel_spec': sample['mel_spec'],
            'token_durations': sample['token_durations']
        }

def collate_batch(batch):
    """Collate function that ensures fixed sequence lengths"""
    # Get max lengths with upper bound
    max_emb_len = min(max(s['embeddings'].size(0) for s in batch), 32)  # Cap at 32
    max_mel_len = max(s['mel_spec'].size(-1) for s in batch)
    
    # Pad sequences
    padded_samples = []
    for sample in batch:
        # Handle embeddings
        embeddings = sample['embeddings']
        if embeddings.size(0) > max_emb_len:
            # Truncate if too long
            embeddings = embeddings[:max_emb_len]
        else:
            # Pad if too short
            emb_pad = max_emb_len - embeddings.size(0)
            embeddings = F.pad(embeddings, (0, 0, 0, emb_pad))
        
        # Handle mel spectrograms
        mel_spec = sample['mel_spec']
        mel_pad = max_mel_len - mel_spec.size(-1)
        mel_spec = F.pad(mel_spec, (0, mel_pad))
        
        # Handle durations
        durations = sample['token_durations']
        if durations.size(0) > max_emb_len:
            durations = durations[:max_emb_len]
        else:
            durations = F.pad(durations, (0, max_emb_len - durations.size(0)))
        
        # Create mask
        mask = torch.ones(max_emb_len, dtype=torch.bool, device=embeddings.device)
        mask[min(sample['embeddings'].size(0), max_emb_len):] = False
        
        padded_samples.append({
            'embeddings': embeddings,
            'mel_spec': mel_spec,
            'token_durations': durations,
            'mask': mask
        })
    
    # Stack into tensors
    batch_dict = {
        'embeddings': torch.stack([s['embeddings'] for s in padded_samples]),
        'mel_spec': torch.stack([s['mel_spec'] for s in padded_samples]),
        'token_durations': torch.stack([s['token_durations'] for s in padded_samples]),
        'mask': torch.stack([s['mask'] for s in padded_samples])
    }
    
    return batch_dict
