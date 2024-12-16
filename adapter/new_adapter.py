import torch
import torch.nn as nn
import torch.nn.functional as F
from f5.modules import (
    DiTBlock, 
    TimestepEmbedding,
    ConvPositionEmbedding,
    AdaLayerNormZero
)

class CSMFusionLayer(nn.Module):
    """
    Implements Canonical Spectral Merge operation for fusing LLaMA and TTS weights
    """
    def __init__(self, dim_out, dim_in):
        super().__init__()
        self.dim_out = dim_out  # tts_dim
        self.dim_in = dim_in    # llama_dim
        
        # Initialize canonical transforms P and Q via SVD of a reference matrix
        W0_tts = torch.randn(dim_out, dim_out)  # [tts_dim x tts_dim]
        W0_llama = torch.randn(dim_in, dim_in)  # [llama_dim x llama_dim]
        
        # Compute SVD for both spaces
        U_tts, _, _ = torch.linalg.svd(W0_tts, full_matrices=False)
        _, _, Vt_llama = torch.linalg.svd(W0_llama, full_matrices=False)
        
        # Store canonical transforms as buffers (non-trainable)
        self.register_buffer('P', U_tts)  # [tts_dim x tts_dim] 
        self.register_buffer('Q', Vt_llama.T)  # [llama_dim x llama_dim]
        
        # Learnable mask for selective spectral combination
        self.spectral_mask = nn.Parameter(torch.ones(dim_out, dim_out))  # [tts_dim x tts_dim]
        
        # Add projection matrices to handle dimension mismatch
        self.proj_down = nn.Linear(dim_in, dim_out)  # Project from llama_dim to tts_dim
        
    def forward(self, x, weights):
        """
        Fuse input features with weight matrix using CSM operation
        Args:
            x: Input features [B x tts_dim]
            weights: Weight matrix [tts_dim x llama_dim]
        Returns:
            Fused features [B x tts_dim]
        """
        batch_size = x.size(0)
        
        # Project weights to TTS space
        w_proj = self.proj_down(weights)  # [tts_dim x tts_dim]
        
        # Move to canonical space
        x_prime = self.P.T @ x.unsqueeze(-1)  # [B x tts_dim x 1]
        w_prime = self.P.T @ w_proj @ self.P  # [tts_dim x tts_dim]
        
        # Apply learnable mask to control fusion
        # Expand w_prime to match batch dimension
        w_prime = w_prime.unsqueeze(0).expand(batch_size, -1, -1)  # [B x tts_dim x tts_dim]
        
        # Apply mask
        masked_w = w_prime * torch.sigmoid(self.spectral_mask).unsqueeze(0)  # [B x tts_dim x tts_dim]
        
        # Fuse with input features
        fused = torch.bmm(masked_w, x_prime)  # [B x tts_dim x 1]
        
        # Move back to original space
        fused = self.P @ fused  # [B x tts_dim x 1]
        
        # Print shapes for debugging
        print(f"x shape: {x.shape}")
        print(f"w_proj shape: {w_proj.shape}")
        print(f"x_prime shape: {x_prime.shape}")
        print(f"w_prime shape: {w_prime.shape}")
        print(f"masked_w shape: {masked_w.shape}")
        print(f"fused pre-squeeze shape: {fused.shape}")
        
        # Ensure output has correct shape
        fused = fused.squeeze(-1)  # [B x tts_dim]
        print(f"fused final shape: {fused.shape}")
        
        # Verify final shape
        assert fused.shape == (batch_size, self.dim_out), f"Expected shape {(batch_size, self.dim_out)}, got {fused.shape}"
        
        return fused

class CustomPositionEmbedding(nn.Module):
    """
    Custom position embedding that matches F5's implementation but handles our dimensions
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=16),
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=1, groups=16)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B x T x tts_dim]
        Returns:
            Position-encoded tensor [B x T x tts_dim]
        """
        
        # Get input dimensions
        B, T, C = x.shape
        assert C == self.dim, f"Expected {self.dim} channels, got {C}"
        
        # Reshape for conv1d
        x = x.permute(0, 2, 1)  # [B x tts_dim x T]
        
        # Apply convolutions
        x = self.conv1d(x)  # [B x tts_dim x T]
        
        # Reshape back
        x = x.permute(0, 2, 1)  # [B x T x tts_dim]
        
        return x

class CSMAdapter(nn.Module):
    """
    Enhanced adapter using Canonical Spectral Merge for weight fusion
    """
    def __init__(
        self,
        llama_dim=3072,
        tts_dim=1024,
        depth=8,
        heads=8,
        dim_head=64,
        n_mel_channels=100
    ):
        super().__init__()
        
        # Store dimensions
        self.llama_dim = llama_dim
        self.tts_dim = tts_dim
        
        # Initialize CSM fusion layer
        self.csm_fusion = CSMFusionLayer(tts_dim, llama_dim)
        
        # Input projection using CSM
        self.input_proj = nn.Linear(llama_dim, tts_dim)
        
        # Time embedding
        self.time_embed = TimestepEmbedding(dim=tts_dim)
        
        # Position encoding - use custom implementation
        self.pos_embedding = CustomPositionEmbedding(tts_dim)
        
        # Processing blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm': AdaLayerNormZero(tts_dim),
                'dit': DiTBlock(
                    dim=tts_dim,
                    heads=heads,
                    dim_head=dim_head
                )
            }) for _ in range(depth)
        ])
        
        # Output layers
        self.final_norm = nn.LayerNorm(tts_dim)
        self.to_mel = nn.Linear(tts_dim, n_mel_channels)
        
        # Duration predictor with better stability
        self.duration_predictor = nn.Sequential(
            nn.LayerNorm(tts_dim),
            nn.Linear(tts_dim, tts_dim // 2),
            nn.LayerNorm(tts_dim // 2),
            nn.ReLU(),
            nn.Linear(tts_dim // 2, 1),
            nn.Softplus()
        )
        
        self.duration_loss_scale = 0.1  # Add duration loss scale parameter
        
    def forward(self, llama_embeddings, timesteps, mask=None, return_durations=False):
        """
        Forward pass with CSM fusion
        Args:
            llama_embeddings: Input embeddings from LLaMA [B x T x llama_dim]
            timesteps: Timestep embeddings [B]
            mask: Attention mask [B x T]
            return_durations: Whether to return duration predictions
        Returns:
            mel_spec: Generated mel spectrogram [B x n_mel x T_expanded]
            durations: (optional) Duration predictions [B x T]
        """
        # Handle default masking
        if mask is None:
            mask = torch.ones(llama_embeddings.shape[:2], dtype=torch.bool, device=llama_embeddings.device)
        else:
            mask = mask.bool()
        
        # Get time embeddings
        time_emb = self.time_embed(timesteps)
        
        batch_size, seq_len, _ = llama_embeddings.shape
        
        # First do regular projection
        x = self.input_proj(llama_embeddings)  # [B x T x tts_dim]
        
        # Predict durations early to influence mel spectrogram generation
        durations = self.duration_predictor(x).squeeze(-1)  # [B x T]
        
        # Expand embeddings based on predicted durations
        x, expanded_mask = self.expand_embeddings(x, durations, mask)  # [B x T_expanded x tts_dim], [B x T_expanded]
        
        # Apply CSM fusion for each position in sequence
        fused_features = []
        for t in range(x.size(1)):
            fused_t = self.csm_fusion(
                x[:, t],  # [B x tts_dim]
                self.input_proj.weight  # [tts_dim x llama_dim]
            )  # [B x tts_dim]
            fused_features.append(fused_t)
        
        x = torch.stack(fused_features, dim=1)  # [B x T_expanded x tts_dim]
        
        # Add positional encoding
        x = self.pos_embedding(x)  # [B x T_expanded x tts_dim]
        
        # Process through blocks
        for block in self.blocks:
            # Apply layer norm with time embedding
            normed, _, _ = block['norm'](x, time_emb)
            
            # Apply DiT block with expanded mask
            x = block['dit'](normed, time_emb, mask=expanded_mask)
        
        # Final processing
        x = self.final_norm(x)
        
        # Generate mel spectrogram after processing
        mel = self.to_mel(x)  # [B x T_expanded x n_mel]
        mel = mel.transpose(1, 2)  # [B x n_mel x T_expanded]
        
        if return_durations:
            return mel, durations
        return mel

    def set_reference_weights(self, reference_weights):
        """
        Set reference weights for CSM fusion basis
        Args:
            reference_weights: Well-trained reference weights [d_out x d_in]
        """
        # Compute SVD of reference weights
        U, S, Vt = torch.linalg.svd(reference_weights, full_matrices=False)
        
        # Update canonical transforms
        self.csm_fusion.P.copy_(U)
        self.csm_fusion.Q.copy_(Vt.T) 

    def expand_embeddings(self, x, durations, mask=None):
        """
        Expand embeddings and masks based on predicted durations.
        Similar to original F5-TTS masking but adapted for duration-based expansion.
        """
        B, T, C = x.size()
        expanded = []
        expanded_masks = []
        
        # Clamp and prepare durations
        durations = torch.clamp(durations, min=1).long()  # [B x T]
        expanded_lengths = durations.sum(dim=-1)  # [B]
        max_len = expanded_lengths.max().item()
        
        # Create position indices for mask creation (similar to original lens_to_mask)
        pos_indices = torch.arange(max_len, device=x.device)[None, :]  # [1 x max_len]  # Not used!
        
        for b in range(B):
            emb_b = x[b]  # [T x C]
            dur_b = durations[b]  # [T]
            emb_expanded_b = emb_b.repeat_interleave(dur_b, dim=0)  # [T_expanded x C]
            
            # Pad to max_len
            pad_len = max_len - emb_expanded_b.size(0)
            if pad_len > 0:
                pad = torch.zeros(pad_len, C, device=x.device)
                emb_expanded_b = torch.cat([emb_expanded_b, pad], dim=0)
            expanded.append(emb_expanded_b)
            
            if mask is not None:
                mask_b = mask[b]  # [T]
                # Create cumulative duration indices for masking
                dur_cumsum = torch.cumsum(dur_b, dim=0)  # [T]
                dur_cumsum_prev = torch.cat([torch.zeros(1, device=x.device), dur_cumsum[:-1]])  # [T]
                
                # Create expanded mask using position-based logic
                mask_expanded_b = torch.zeros(max_len, dtype=torch.bool, device=mask.device)
                for t, (start, end) in enumerate(zip(dur_cumsum_prev, dur_cumsum)):
                    if mask_b[t]:  # Only expand valid positions
                        mask_expanded_b[int(start):int(end)] = True
                expanded_masks.append(mask_expanded_b)
        
        expanded_x = torch.stack(expanded, dim=0)  # [B x max_len x tts_dim]
        expanded_mask = torch.stack(expanded_masks, dim=0) if mask is not None else None  # [B x max_len]
        
        # Validate expanded mask (similar to original F5-TTS validation)
        if expanded_mask is not None:
            assert expanded_mask.dtype == torch.bool, f"Expected bool mask, got {expanded_mask.dtype}"
            assert expanded_mask.shape == (B, max_len), f"Expected mask shape {(B, max_len)}, got {expanded_mask.shape}"
            assert expanded_mask.sum(dim=1).min() > 0, "Found sequence with no valid positions"
        
        return expanded_x, expanded_mask

    def expand_mel(self, mel_spec, token_durations):
        """
        Expand mel spectrogram based on token durations.
        Args:
            mel_spec: Target mel spectrogram [B x n_mel x T]
            token_durations: Ground truth durations [B x T_tokens]
        Returns:
            Expanded mel spectrogram [B x n_mel x T_expanded]
        """
        B, n_mel, T = mel_spec.shape
        durations = token_durations.int()
        
        # Handle duration length mismatch
        if durations.size(1) > T:
            durations = durations[:, :T]
        elif durations.size(1) < T:
            pad_size = T - durations.size(1)
            duration_pad = torch.ones(B, pad_size, device=durations.device, dtype=durations.dtype)
            durations = torch.cat([durations, duration_pad], dim=1)
        
        # Ensure no zero durations
        durations = torch.clamp(durations, min=1)
        
        mel_expanded = []
        for b in range(B):
            mel_b = mel_spec[b]  # [n_mel x T]
            dur_b = durations[b]  # [T]
            mel_expanded_b = mel_b.repeat_interleave(dur_b, dim=1)  # [n_mel x T_expanded]
            mel_expanded.append(mel_expanded_b)
        
        # Find maximum length for padding
        max_len = max([mel.size(1) for mel in mel_expanded])
        
        # Pad each expanded mel to max length
        padded_mels = []
        for mel in mel_expanded:
            if mel.size(1) < max_len:
                pad_size = max_len - mel.size(1)
                pad = torch.zeros(n_mel, pad_size, device=mel.device)
                mel_padded = torch.cat([mel, pad], dim=1)
            else:
                mel_padded = mel
            padded_mels.append(mel_padded)
        
        mel_expanded = torch.stack(padded_mels, dim=0)  # [B x n_mel x T_expanded]
        return mel_expanded