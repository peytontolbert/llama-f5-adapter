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
    FeedForward,
    SinusPositionEmbedding
)
import math

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
        
        # Increase dropout for better regularization
        self.dropout = nn.Dropout(dropout * 2)
        
        # Add layer normalization before projection
        self.input_norm = nn.LayerNorm(llama_dim)
        
        # Add weight decay for regularization
        self.weight_decay = 0.01
        
        # Initial projection with better normalization
        self.input_proj = nn.Sequential(
            self.input_norm,
            nn.Linear(llama_dim, tts_dim),
            nn.LayerNorm(tts_dim),
            nn.ReLU(),
            GRN(tts_dim),
            self.dropout
        )
        
        # Positional embeddings with consistent dimensions
        self.pos_embed = ConvPositionEmbedding(tts_dim)
        self.rel_pos = RelativePositionalEmbedding(
            dim=tts_dim,
            num_heads=heads,
            dim_head=dim_head,
            max_positions=2048
        )
        
        # Time embedding
        self.time_embed = TimestepEmbedding(tts_dim)
        
        # Main transformer blocks with consistent dimensions
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=tts_dim,
                heads=heads,
                dim_head=dim_head,  # Pass the same dim_head throughout
                ff_mult=ff_mult,
                use_relative_pos=True
            ) for _ in range(depth)
        ])
        
        # Final normalization and projection - Remove Sequential
        self.final_norm = AdaLayerNormZero(tts_dim)  # Change back to AdaLayerNormZero
        
        # Mel spectrogram projection with better range handling
        self.to_mel = nn.Sequential(
            FeedForward(tts_dim, tts_dim // 2, dropout=dropout),
            nn.LayerNorm(tts_dim // 2),
            nn.GELU(),
            nn.Linear(tts_dim // 2, n_mel_channels),
            nn.Tanh()
        )
        
        # Store dimensions for shape checking
        self.n_mel_channels = n_mel_channels
        self.tts_dim = tts_dim
        self.min_seq_len = min_seq_len
        
        # Expand hidden states based on predicted durations
        self.expand_states = nn.Sequential(
            nn.Linear(tts_dim, tts_dim * 4),
            nn.LayerNorm(tts_dim * 4),
            nn.GELU(),
            nn.Linear(tts_dim * 4, tts_dim)
        )
        
        # Temporal projection
        self.temporal_proj = nn.Linear(tts_dim, tts_dim)
        
        # Single duration predictor with proper architecture
        self.duration_predictor = DurationPredictor(
            hidden_dim=tts_dim,
            input_dim=llama_dim
        )
        
        # Separate mel normalizer
        self.mel_normalizer = MelRangeNormalization(n_mel_channels)
        
        # Add sequence length control
        self.max_duration_per_token = 32
        self.target_duration_per_token = 24  # F5-TTS average
        
        # Refined duration control
        self.chunk_size = 512
        self.frames_per_token = 24  # Target F5-TTS average
        self.min_duration = 8
        self.max_duration = 32
        
        # Add gradient clipping threshold
        self.grad_clip_thresh = 1.0
        
        # Add weight initialization
        self._init_weights()
        
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def _init_weights(self):
        """Initialize weights with better scaling"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Special initialization for adapter layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use smaller initialization for adapter layers
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def predict_chunk_durations(self, chunk, target_length=None):
        """Predict and adjust durations for a chunk of tokens"""
        # Get initial duration predictions
        duration_pred = self.duration_predictor(chunk)  # [B, T, 1]
        durations = duration_pred.squeeze(-1)  # [B, T]
        
        # If no target length specified, use frames_per_token average
        if target_length is None:
            target_length = chunk.size(1) * self.frames_per_token
            
        # Scale durations to match target length while preserving relative durations
        for b in range(durations.size(0)):
            current_sum = durations[b].sum()
            scale = target_length / current_sum
            durations[b] = (durations[b] * scale).round()
            
            # Clamp individual durations
            durations[b] = durations[b].clamp(min=self.min_duration, max=self.max_duration)
            
            # Fine-tune to exactly match target length
            current_sum = durations[b].sum()
            diff = target_length - current_sum
            
            if diff > 0:
                # Add frames to shortest durations
                for _ in range(int(diff)):
                    idx = durations[b].argmin()
                    if durations[b, idx] < self.max_duration:
                        durations[b, idx] += 1
            elif diff < 0:
                # Remove frames from longest durations
                for _ in range(int(-diff)):
                    idx = durations[b].argmax()
                    if durations[b, idx] > self.min_duration:
                        durations[b, idx] -= 1
        
        return durations.long()
    
    def forward(self, llama_embeddings, timesteps, mask=None, return_features=False, ref_audio_len=None, ref_text=None, speed=1.0, return_durations=False):
        """Process embeddings similar to F5-TTS text processing"""
        # Ensure inputs are on correct device
        device = llama_embeddings.device
        if timesteps.device != device:
            timesteps = timesteps.to(device)
        if mask is not None and mask.device != device:
            mask = mask.to(device)
        
        batch_size = llama_embeddings.shape[0]
        
        # Project embeddings
        projected_embeddings = self.input_proj(llama_embeddings)  # [B, T, tts_dim]
        
        # Duration prediction and scaling
        duration_pred = self.duration_predictor(llama_embeddings)
        durations = (duration_pred.squeeze(-1) * speed).round().long()
        durations = durations.clamp(min=self.min_duration, max=self.max_duration)
        
        # Calculate max length after expansion
        max_expanded_len = max([
            sum([int(durations[b, t].item()) for t in range(llama_embeddings.size(1))
                if mask is None or mask[b, t]])
            for b in range(batch_size)
        ])
        
        # Expand states based on predicted durations with padding
        expanded_sequences = []
        for b in range(batch_size):
            expanded_seq = []
            current_len = 0
            
            for t in range(llama_embeddings.size(1)):
                if mask is None or mask[b, t]:
                    feat = self.expand_states(projected_embeddings[b, t])
                    dur = int(durations[b, t].item())
                    expanded = feat.unsqueeze(0).expand(dur, -1)
                    expanded_seq.append(expanded)
                    current_len += dur
            
            # Concatenate and pad if necessary
            expanded_seq = torch.cat(expanded_seq, dim=0)  # [L, D]
            if current_len < max_expanded_len:
                # Create padding tensor on same device as expanded_seq
                padding = torch.zeros(
                    max_expanded_len - current_len, 
                    self.tts_dim, 
                    device=device  # Use input device
                )
                expanded_seq = torch.cat([expanded_seq, padding], dim=0)
            
            expanded_sequences.append(expanded_seq)
        
        hidden_states = torch.stack(expanded_sequences, dim=0)  # [B, T_mel, tts_dim]
        
        # Better feature mixing
        x = self.pos_embed(hidden_states)
        rel_pos = self.rel_pos(x)  # Store relative position embeddings
        if rel_pos is not None:
            x = x + self.dropout(rel_pos)
        
        # Get time embeddings
        time_emb = self.time_embed(timesteps)
        
        # Layer-wise feature normalization
        layer_norms = []
        for i in range(len(self.transformer_blocks)):
            layer_norms.append(nn.LayerNorm(self.tts_dim).to(x.device))
        
        # Create mel-level mask based on expanded sequence lengths
        mel_mask = None
        if mask is not None:
            mel_lengths = [
                sum([int(durations[b, t].item()) for t in range(llama_embeddings.size(1))
                    if mask[b, t]])
                for b in range(batch_size)
            ]
            mel_mask = torch.arange(max_expanded_len, device=x.device)[None, :] < torch.tensor(mel_lengths, device=x.device)[:, None]
        
        # Process through transformer blocks with relative positions and mask
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, time_emb, rel_pos=rel_pos, mask=mel_mask)
            x = layer_norms[i](x)
        
        # Final normalization and mel projection
        x, scale, shift = self.final_norm(x, time_emb)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # Generate mel spectrogram
        mel_output = self.to_mel(x)  # [B, T, n_mel_channels]
        mel_output = mel_output.transpose(1, 2)  # [B, n_mel_channels, T]
        
        # Apply mel range normalization
        mel_output = self.mel_normalizer(mel_output)
        
        if return_durations:
            return mel_output, duration_pred
        return mel_output

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
        self.register_buffer('target_min', torch.tensor(-6.0))
        self.register_buffer('target_max', torch.tensor(6.0))
        self.register_buffer('target_mean', torch.tensor(-0.86))
        self.register_buffer('target_std', torch.tensor(2.5))
        
        # Add learnable scaling factors
        self.scale = nn.Parameter(torch.ones(1, n_mel_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, n_mel_channels, 1))
        
    def forward(self, x):
        # Normalize to zero mean and unit variance
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-5
        x = (x - mean) / std
        
        # Apply learnable scaling
        x = x * self.scale + self.bias
        
        # Scale to target distribution
        x = x * self.target_std + self.target_mean
        
        # Clip to target range
        x = torch.clamp(x, self.target_min, self.target_max)
        return x

class DurationPredictor(nn.Module):
    def __init__(self, hidden_dim, input_dim=3072):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Replace fixed positional embedding with sinusoidal
        self.register_buffer(
            'position_ids',
            torch.arange(2048).expand((1, -1))  # Increased max length
        )
        
        # Use the RelativePositionalEmbedding from f5.modules
        self.rel_pos = RelativePositionalEmbedding(
            dim=hidden_dim,
            num_heads=1,  # Single head for duration prediction
            dim_head=hidden_dim,
            max_positions=2048
        )
        
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(hidden_dim, 1)
        
        # Add sinusoidal position encoding
        self.pos_encoding = SinusPositionEmbedding(hidden_dim)
        
    def forward(self, x):
        # x: [B, T, H]
        batch_size, seq_len = x.shape[:2]
        
        # Project input
        x = self.input_proj(x)
        x = self.norm1(x)
        
        # Add sinusoidal positional embeddings
        positions = torch.arange(seq_len, device=x.device).float()
        pos_emb = self.pos_encoding(positions)  # [T, H]
        x = x + pos_emb.unsqueeze(0)  # Add to batch dimension
        
        # Get relative positional bias
        rel_pos = self.rel_pos(x)  # [B, H, T, T]
        
        # Handle case where rel_pos is None
        if rel_pos is not None:
            # Extract diagonal elements for position-wise features
            rel_pos_diag = torch.diagonal(rel_pos, dim1=-2, dim2=-1)  # [B, H, T]
            rel_pos_features = rel_pos_diag.permute(0, 2, 1)  # [B, T, H]
            
            # Project relative position features to match hidden dimension
            rel_pos_features = F.linear(
                rel_pos_features,
                torch.eye(x.shape[-1], device=x.device)[:rel_pos_features.shape[-1]]
            )
            
            x = x + rel_pos_features
        
        x = self.dropout(x)
        
        # Convolutional processing
        x_conv = x.transpose(1, 2)  # [B, H, T]
        x_conv = F.relu(self.conv1(x_conv))
        x_conv = x_conv.transpose(1, 2)  # [B, T, H]
        x_conv = self.norm2(x_conv)
        
        x_conv = x_conv.transpose(1, 2)  # [B, H, T]
        x_conv = F.relu(self.conv2(x_conv))
        x_conv = x_conv.transpose(1, 2)  # [B, T, H]
        x_conv = self.norm3(x_conv)
        
        # Final projection with positive output
        durations = F.softplus(self.proj(x_conv))  # [B, T, 1]
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
