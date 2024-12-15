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
        
        # Duration predictor
        self.duration_predictor = nn.Sequential(
            nn.Linear(tts_dim, tts_dim // 2),
            nn.ReLU(),
            nn.Linear(tts_dim // 2, 1),
            nn.Softplus()
        )
        
    def forward(self, llama_embeddings, timesteps, mask=None, return_durations=False):
        """
        Forward pass with CSM fusion
        Args:
            llama_embeddings: Input embeddings from LLaMA [B x T x llama_dim]
            timesteps: Timestep embeddings [B]
            mask: Attention mask [B x T]
            return_durations: Whether to return duration predictions
        Returns:
            mel_spec: Generated mel spectrogram [B x n_mel x T]
            durations: (optional) Duration predictions [B x T]
        """
        # Get time embeddings
        time_emb = self.time_embed(timesteps)
        
        batch_size, seq_len, _ = llama_embeddings.shape
        
        # First do regular projection
        x = self.input_proj(llama_embeddings)  # [B x T x tts_dim]
        
        # Apply CSM fusion for each position in sequence
        fused_features = []
        for t in range(seq_len):
            
            fused_t = self.csm_fusion(
                x[:, t],  # [B x tts_dim]
                self.input_proj.weight  # [tts_dim x llama_dim]
            )  # [B x tts_dim]
            
            
            fused_features.append(fused_t)
        
        # Stack fused features along sequence dimension
        x = torch.stack(fused_features, dim=1)  # [B x T x tts_dim]
        # Verify dimensions
        assert x.shape == (batch_size, seq_len, self.tts_dim), f"Expected shape {(batch_size, seq_len, self.tts_dim)}, got {x.shape}"
        
        # Add positional encoding
        x = self.pos_embedding(x)  # [B x T x tts_dim]
        
        # Process through blocks
        for block in self.blocks:
            # Apply layer norm with time embedding
            normed, _, _ = block['norm'](x, time_emb)
            
            # Apply DiT block
            x = block['dit'](normed, time_emb, mask=mask)
        
        # Final processing
        x = self.final_norm(x)
        
        # Predict durations
        durations = self.duration_predictor(x).squeeze(-1)  # [B x T]
        
        # Generate mel spectrogram
        mel = self.to_mel(x)  # [B x T x n_mel]
        mel = mel.transpose(1, 2)  # [B x n_mel x T]
        
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