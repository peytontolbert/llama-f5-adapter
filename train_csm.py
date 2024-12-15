import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from adapter.new_adapter import CSMAdapter
from tqdm import tqdm
import math

class CSMTrainer:
    def __init__(
        self,
        adapter,
        train_dataset,
        val_dataset=None,
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=100,
        device='cuda'
    ):
        self.adapter = adapter.to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_batch
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_batch
        ) if val_dataset else None
        
        # Optimizer with separate parameter groups
        self.optimizer = torch.optim.AdamW([
            {'params': [p for n, p in adapter.named_parameters() if 'csm_fusion' in n],
             'lr': learning_rate * 0.1},  # Lower LR for CSM parameters
            {'params': [p for n, p in adapter.named_parameters() if 'csm_fusion' not in n],
             'lr': learning_rate}
        ])
        
        # Use OneCycleLR for better convergence
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,  # Shorter warmup
            div_factor=25,  # Lower starting LR
            final_div_factor=1000,
            anneal_strategy='cos'
        )
        
        # Initialize EMA model
        self.ema = torch.optim.swa_utils.AveragedModel(
            adapter,
            avg_fn=lambda avg, new, num: 0.999 * avg + 0.001 * new
        )
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Add loss scaling factors
        self.mel_loss_scale = 0.01  # Scale down mel losses
        self.duration_loss_scale = 0.0001  # Scale down duration losses
        
    def collate_batch(self, batch):
        """Custom collate function for batching data"""
        # Get max lengths
        max_emb_len = min(max(s['embeddings'].size(0) for s in batch), 32)
        max_mel_len = max(s['mel_spec'].size(-1) for s in batch)
        
        # Initialize tensors
        embeddings = torch.zeros(len(batch), max_emb_len, batch[0]['embeddings'].size(-1))
        mel_specs = torch.zeros(len(batch), batch[0]['mel_spec'].size(0), max_mel_len)
        durations = torch.zeros(len(batch), max_emb_len)
        masks = torch.zeros(len(batch), max_emb_len, dtype=torch.bool)
        
        # Fill tensors
        for i, sample in enumerate(batch):
            emb_len = min(sample['embeddings'].size(0), max_emb_len)
            mel_len = sample['mel_spec'].size(-1)
            
            embeddings[i, :emb_len] = sample['embeddings'][:emb_len]
            mel_specs[i, :, :mel_len] = sample['mel_spec'][:, :mel_len]
            
            # Get durations from alignment info
            if 'alignment' in sample and 'token_durations' in sample['alignment']:
                dur_len = min(len(sample['alignment']['token_durations']), emb_len)
                durations[i, :dur_len] = sample['alignment']['token_durations'][:dur_len]
            else:
                # Default duration if not available (8 frames per token)
                durations[i, :emb_len] = 8
                
            masks[i, :emb_len] = True
            
        return {
            'embeddings': embeddings.to(self.device),
            'mel_spec': mel_specs.to(self.device),
            'token_durations': durations.to(self.device),
            'mask': masks.to(self.device)
        }
        
    def train_epoch(self, epoch):
        self.adapter.train()
        total_loss = 0
        mel_losses = []
        duration_losses = []
        
        progress = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(progress):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                pred_mel, pred_durations = self.adapter(
                    batch['embeddings'],
                    timesteps=torch.zeros(len(batch['embeddings']), device=self.device),
                    mask=batch['mask'],
                    return_durations=True
                )
                
                # Compute losses
                mel_loss = self.compute_mel_loss(pred_mel, batch['mel_spec'], batch['mask'])
                duration_loss = self.duration_loss_scale * self.compute_duration_loss(
                    pred_durations,
                    batch['token_durations'],
                    batch['mask']
                )
                
                loss = mel_loss + duration_loss
            
            # Gradient clipping before scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
            
            # Optimizer and scheduler steps
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # Update EMA model
            self.ema.update_parameters(self.adapter)
            
            # Track losses
            mel_losses.append(mel_loss.item())
            duration_losses.append(duration_loss.item())
            total_loss += loss.item()
            
            # Update progress bar
            progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mel_loss': f"{mel_loss.item():.4f}",
                'dur_loss': f"{duration_loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_mel_loss = sum(mel_losses) / len(mel_losses)
        avg_dur_loss = sum(duration_losses) / len(duration_losses)
        
        print(f"\nEpoch {epoch} averages:")
        print(f"Total Loss: {avg_loss:.4f}")
        print(f"Mel Loss: {avg_mel_loss:.4f}")
        print(f"Duration Loss: {avg_dur_loss:.4f}")
        
        return avg_loss
    
    def compute_mel_loss(self, pred_mel, target_mel, mask):
        """Compute mel spectrogram loss with masking and better normalization"""
        # Normalize mel specs to [-1, 1] range
        pred_mel = 2 * (pred_mel - pred_mel.min()) / (pred_mel.max() - pred_mel.min() + 1e-8) - 1
        target_mel = 2 * (target_mel - target_mel.min()) / (target_mel.max() - target_mel.min() + 1e-8) - 1
        
        # Create mel mask
        mel_len = min(pred_mel.size(-1), target_mel.size(-1))
        token_len = mask.size(-1)
        
        # Interpolate mask to mel length
        mel_mask = F.interpolate(
            mask.unsqueeze(1).float(),
            size=mel_len,
            mode='nearest'
        )
        
        # Truncate to same length
        pred_mel = pred_mel[..., :mel_len]
        target_mel = target_mel[..., :mel_len]
        
        # L1 loss with better normalization
        l1_loss = (torch.abs(pred_mel - target_mel) * mel_mask).sum() / (mel_mask.sum() + 1e-8)
        
        # MSE loss with better normalization
        mse_loss = (torch.pow(pred_mel - target_mel, 2) * mel_mask).sum() / (mel_mask.sum() + 1e-8)
        
        # Add spectral convergence loss
        spec_loss = self.spectral_convergence_loss(pred_mel, target_mel, mel_mask)
        
        return self.mel_loss_scale * (l1_loss + 0.1 * mse_loss + 0.1 * spec_loss)
    
    def spectral_convergence_loss(self, pred_mel, target_mel, mask):
        """Compute spectral convergence loss"""
        pred_fft = torch.fft.rfft(pred_mel * mask, dim=-1)
        target_fft = torch.fft.rfft(target_mel * mask, dim=-1)
        
        return torch.norm(torch.abs(target_fft) - torch.abs(pred_fft), p='fro') / (torch.norm(torch.abs(target_fft), p='fro') + 1e-8)
    
    def compute_duration_loss(self, pred_durations, target_durations, mask):
        """Compute duration prediction loss"""
        # Basic MSE loss
        duration_loss = F.mse_loss(
            pred_durations * mask.float(),
            target_durations * mask.float(),
            reduction='sum'
        ) / (mask.sum() + 1e-8)
        
        # Relative duration loss
        pred_total = (pred_durations * mask.float()).sum(dim=1) + 1e-8
        target_total = (target_durations * mask.float()).sum(dim=1) + 1e-8
        
        relative_loss = F.l1_loss(
            torch.log(pred_total / pred_total.mean()),
            torch.log(target_total / target_total.mean())
        )
        
        return duration_loss + 0.1 * relative_loss
    
    def validate(self):
        """Validate model on validation set"""
        if not self.val_loader:
            return 0.0
            
        self.ema.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Forward pass
                pred_mel, pred_durations = self.ema(
                    batch['embeddings'],
                    timesteps=torch.zeros(len(batch['embeddings']), device=self.device),
                    mask=batch['mask'],
                    return_durations=True
                )
                
                # Interpolate predicted mel
                pred_mel = F.interpolate(
                    pred_mel,
                    size=batch['mel_spec'].size(-1),
                    mode='linear',
                    align_corners=False
                )
                
                # Compute losses
                mel_loss = self.compute_mel_loss(pred_mel, batch['mel_spec'], batch['mask'])
                duration_loss = self.compute_duration_loss(
                    pred_durations,
                    batch['token_durations'],
                    batch['mask']
                )
                
                total_loss += mel_loss + 0.1 * duration_loss
                
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Print metrics
            print(f'\nEpoch {epoch}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.adapter.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss
                }, 'best_csm_adapter.pt') 

def main():
    import pickle
    
    print("Loading dataset...")
    try:
        with open("generated_dataset.pkl", 'rb') as f:
            dataset = pickle.load(f)
            if isinstance(dataset, dict):
                dataset = dataset['data']
    except FileNotFoundError:
        print("Error: Could not find generated_dataset.pkl")
        print("Please run generate_dataset.py first")
        return
        
    print(f"Loaded {len(dataset)} samples")
    
    # Split into train/val
    from torch.utils.data import random_split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print("Initializing CSM Adapter...")
    adapter = CSMAdapter(
        llama_dim=3072,
        tts_dim=1024,
        depth=8,
        heads=8,
        dim_head=64,
        n_mel_channels=100
    )
    
    print(f"Total parameters: {sum(p.numel() for p in adapter.parameters())}")
    
    trainer = CSMTrainer(
        adapter=adapter,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=100
    )
    
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()