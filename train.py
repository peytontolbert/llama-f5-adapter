import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from f5.utils_infer import load_vocoder
from f5.utils import lens_to_mask
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
import math
import json
from datetime import datetime
import time

# Configuration
DEVICE = torch.device('cuda:0')
BATCH_SIZE = 2  # Slightly larger batch for better statistics
EPOCHS = 200
LEARNING_RATE = 0.000005  # Reduced from 0.00001
WARMUP_STEPS = 500
GRADIENT_CLIP = 0.01  # Tighter gradient clipping
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
GRADIENT_ACCUMULATION_STEPS = 32  # More accumulation for stability
WEIGHT_DECAY = 0.00001  # Lower weight decay
MIN_LR_RATIO = 0.1
INITIAL_LR_SCALE = 0.1
DURATION_MIN = 1
DURATION_MAX = 50
MEL_LOSS_WEIGHT = 1.0  # Increased from 0.7
MSE_LOSS_WEIGHT = 0.1  # Reduced from 0.3
DURATION_LOSS_WEIGHT = 0.005  # Further reduced from 0.01
WARMUP_EPOCHS = 10  # Longer warmup

class AdapterDataset(Dataset):
    def __init__(self, dataset_path):
        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict) and 'data' in data:
                self.samples = []
                # Process samples to ensure consistent format
                for sample in data['data']:
                    processed = {}
                    processed['embeddings'] = sample['embeddings']
                    processed['mel_spec'] = sample['mel_spec']
                    
                    # Handle token durations
                    if 'token_durations' in sample:
                        processed['token_durations'] = sample['token_durations']
                    elif 'alignment' in sample and 'token_durations' in sample['alignment']:
                        processed['token_durations'] = sample['alignment']['token_durations']
                    else:
                        # Default duration of 8 frames per token
                        processed['token_durations'] = torch.ones(sample['embeddings'].size(0)) * 8
                    
                    self.samples.append(processed)
            else:
                self.samples = data
        print(f"Loaded {len(self.samples)} samples")
        
        # Calculate duration statistics for the dataset
        durations = []
        for sample in self.samples:
            if 'alignment' in sample and 'token_durations' in sample['alignment']:
                durations.extend(sample['alignment']['token_durations'].tolist())
        
        if durations:
            self.token_duration_stats = {
                'mean': np.mean(durations),
                'std': np.std(durations)
            }
            print(f"Duration statistics - Mean: {self.token_duration_stats['mean']:.2f}, Std: {self.token_duration_stats['std']:.2f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Ensure token_durations exists in sample
        if 'token_durations' not in sample:
            if 'alignment' in sample and 'token_durations' in sample['alignment']:
                sample['token_durations'] = sample['alignment']['token_durations']
            else:
                # Create default durations (8 frames per token)
                sample['token_durations'] = torch.ones(sample['embeddings'].size(0)) * 8
        return sample

def collate_batch(batch):
    """Collate function to handle variable length sequences"""
    # Process each sample, potentially creating multiple chunks
    chunked_samples = []
    
    for sample in batch:
        emb_len = sample['embeddings'].size(0)
        mel_len = sample['mel_spec'].size(-1)
        
        # If sequence is longer than max length, chunk it
        max_emb_len = 2048
        if emb_len > max_emb_len:
            # Calculate number of chunks needed
            num_chunks = math.ceil(emb_len / max_emb_len)
            
            for i in range(num_chunks):
                start_idx = i * max_emb_len
                end_idx = min((i + 1) * max_emb_len, emb_len)
                mel_start = start_idx * 8  # Approximate mel start
                mel_end = min(end_idx * 8, mel_len)
                
                chunk = {
                    'embeddings': sample['embeddings'][start_idx:end_idx],
                    'mel_spec': sample['mel_spec'][..., mel_start:mel_end],
                    'token_durations': sample['token_durations'][start_idx:end_idx]
                }
                chunked_samples.append(chunk)
        else:
            chunked_samples.append(sample)
    
    # Find max lengths in the chunked samples
    max_emb_len = max(s['embeddings'].size(0) for s in chunked_samples)
    max_mel_len = max(s['mel_spec'].size(-1) for s in chunked_samples)
    
    # Pad and stack
    padded_embeddings = []
    padded_mel_specs = []
    padded_durations = []
    
    for sample in chunked_samples:
        # Pad embeddings
        emb_len = sample['embeddings'].size(0)
        padded_emb = F.pad(
            sample['embeddings'],
            (0, 0, 0, max_emb_len - emb_len)
        )
        padded_embeddings.append(padded_emb)
        
        # Pad mel spectrograms
        mel_len = sample['mel_spec'].size(-1)
        padded_mel = F.pad(
            sample['mel_spec'],
            (0, max_mel_len - mel_len)
        )
        padded_mel_specs.append(padded_mel)
        
        # Pad durations
        dur_len = sample['token_durations'].size(0)
        padded_dur = F.pad(
            sample['token_durations'],
            (0, max_emb_len - dur_len)
        )
        padded_durations.append(padded_dur)
    
    # Create batch dictionary
    batch_dict = {
        'embeddings': torch.stack(padded_embeddings).to(DEVICE),
        'mel_spec': torch.stack(padded_mel_specs).to(DEVICE),
        'token_durations': torch.stack(padded_durations).to(DEVICE),
        'num_chunks': len(chunked_samples) // len(batch),  # Track number of chunks per original sample
        'is_chunked': len(chunked_samples) > len(batch),
        'emb_lengths': torch.tensor([s['embeddings'].size(0) for s in chunked_samples]).to(DEVICE),
        'mel_lengths': torch.tensor([s['mel_spec'].size(-1) for s in chunked_samples]).to(DEVICE)
    }
    
    return batch_dict

def normalize_mel(mel):
    """Normalize mel spectrogram with robust scaling"""
    B, C, T = mel.size()
    
    # First normalize per channel
    mel_mean = mel.mean(dim=2, keepdim=True)
    mel_std = mel.std(dim=2, keepdim=True) + 1e-5
    mel = (mel - mel_mean) / mel_std
    
    # Then apply tanh for bounded output
    mel = torch.tanh(mel * 0.5)
    
    return mel

def compute_masked_loss(pred, target, mask, loss_fn):
    """Compute loss only on valid regions with improved normalization"""
    # Apply mask
    masked_pred = pred * mask
    masked_target = target * mask
    
    # Get number of valid elements
    valid_elements = mask.sum() + 1e-8
    
    # Compute raw loss
    raw_loss = loss_fn(masked_pred, masked_target)
    
    # Normalize by valid elements
    normalized_loss = raw_loss * mask.numel() / valid_elements
    
    # Add L2 regularization for stability
    l2_reg = 0.0001 * (pred ** 2).mean()
    
    return normalized_loss + l2_reg

def compute_duration_loss(pred_durations, target_durations, attention_mask):
    """Compute duration loss with better normalization"""
    # Get valid elements
    valid_elements = attention_mask.sum() + 1e-8
    
    # Basic MSE loss on durations with proper masking
    duration_loss = F.mse_loss(
        pred_durations * attention_mask.float(),
        target_durations * attention_mask.float(),
        reduction='sum'
    ) / valid_elements
    
    # Relative duration loss with log-space comparison
    pred_total = (pred_durations * attention_mask.float()).sum(dim=1) + 1e-8
    target_total = (target_durations * attention_mask.float()).sum(dim=1) + 1e-8
    
    relative_loss = F.l1_loss(
        torch.log(pred_total),
        torch.log(target_total),
        reduction='mean'
    )
    
    return duration_loss, relative_loss

class AdapterTrainer:
    def __init__(self, adapter, dataset_path, vocoder=None):
        self.adapter = adapter.to(DEVICE)
        self.dataset = AdapterDataset(dataset_path)
        self.vocoder = vocoder
        
        # Initialize logging
        self.log_file = CHECKPOINT_DIR / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.training_history = {
            'config': {
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'epochs': EPOCHS,
                'warmup_steps': WARMUP_STEPS,
                'gradient_clip': GRADIENT_CLIP,
                'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
                'weight_decay': WEIGHT_DECAY,
                'model_parameters': sum(p.numel() for p in self.adapter.parameters()),
                'dataset_size': len(self.dataset)
            },
            'epochs': []
        }
        
        # Save initial config
        self._save_training_log()
        
        # Create separate train/val datasets
        dataset_size = len(self.dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        # Create dataloaders with collate_fn
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_batch
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_batch
        )
        
        # Optimizer setup with separate parameter groups
        self.optimizer = torch.optim.AdamW(
            [
                {'params': [p for n, p in self.adapter.named_parameters() if 'norm' not in n],
                 'weight_decay': WEIGHT_DECAY},
                {'params': [p for n, p in self.adapter.named_parameters() if 'norm' in n],
                 'weight_decay': 0.0}
            ],
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # More conservative scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=LEARNING_RATE,
            epochs=EPOCHS,
            steps_per_epoch=len(self.train_dataloader),
            pct_start=0.2,  # Longer warmup
            div_factor=25,
            final_div_factor=1000,
            anneal_strategy='linear'
        )
        
        # More conservative EMA
        self.ema = torch.optim.swa_utils.AveragedModel(
            self.adapter,
            avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: 
                0.995 * averaged_model_parameter + 0.005 * model_parameter
        )
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=2**10)
    
    def _save_training_log(self):
        """Save training history to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)

    def _log_batch_metrics(self, batch_metrics, epoch, batch_idx):
        """Log detailed batch-level metrics"""
        if len(self.training_history['epochs']) <= epoch:
            self.training_history['epochs'].append({
                'epoch_idx': epoch,
                'batches': [],
                'epoch_metrics': {},
                'validation_metrics': {},
                'convergence_metrics': {},
                'timing': {
                    'start_time': time.time(),
                    'end_time': None,
                    'duration': None
                }
            })
        
        self.training_history['epochs'][epoch]['batches'].append({
            'batch_idx': batch_idx,
            'metrics': batch_metrics
        })

    def _log_epoch_metrics(self, epoch, metrics):
        """Log epoch-level metrics"""
        self.training_history['epochs'][epoch]['epoch_metrics'] = metrics
        self.training_history['epochs'][epoch]['timing']['end_time'] = time.time()
        self.training_history['epochs'][epoch]['timing']['duration'] = (
            self.training_history['epochs'][epoch]['timing']['end_time'] - 
            self.training_history['epochs'][epoch]['timing']['start_time']
        )
        self._save_training_log()

    def get_cosine_schedule_with_warmup(
        self,
        optimizer,
        num_training_steps,
        num_warmup_steps,
        num_cycles=1,
        min_lr_ratio=MIN_LR_RATIO
    ):
        def lr_lambda(current_step):
            # Longer linear warmup
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            # Cosine decay with minimum learning rate
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def normalize_mel(self, mel_spec):
        # Normalize to [-1, 1] range which is standard for mel specs
        min_val = mel_spec.min(dim=-1, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        max_val = mel_spec.max(dim=-1, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        normalized = 2 * (mel_spec - min_val) / (max_val - min_val + 1e-5) - 1
        
        return normalized
    
    def train_epoch(self, epoch):
        self.adapter.train()
        total_loss = 0
        valid_batches = 0
        epoch_losses = {
            'l1_loss': 0.0,
            'mse_loss': 0.0,
            'duration_loss': 0.0,
            'relative_duration_loss': 0.0
        }
        
        progress = tqdm(self.train_dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(progress):
            try:
                # Get current learning rate scale
                lr_scale = self.get_lr_scale(epoch, batch_idx)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = LEARNING_RATE * lr_scale
                
                # Create attention mask
                attention_mask = torch.arange(batch['embeddings'].size(1), device=DEVICE)[None, :] < batch['emb_lengths'][:, None]
                
                # Forward pass with duration prediction
                pred_mel, duration_pred = self.adapter(
                    batch['embeddings'],
                    timesteps=torch.zeros(batch['embeddings'].size(0), device=DEVICE),
                    mask=attention_mask,
                    return_durations=True
                )
                
                # Interpolate to match target length
                pred_mel = F.interpolate(
                    pred_mel,
                    size=batch['mel_spec'].size(-1),
                    mode='linear',
                    align_corners=False
                )
                
                # Create mel-level mask
                mel_mask = torch.arange(batch['mel_spec'].size(-1), device=DEVICE)[None, :] < batch['mel_lengths'][:, None]
                mel_mask = mel_mask.unsqueeze(1).float()
                
                # Normalize mels
                pred_mel = normalize_mel(pred_mel)
                target_mel = normalize_mel(batch['mel_spec'])
                
                # Compute mel losses
                l1_loss = compute_masked_loss(
                    pred_mel, target_mel, mel_mask,
                    lambda x, y: F.l1_loss(x, y, reduction='mean')
                )
                
                mse_loss = compute_masked_loss(
                    pred_mel, target_mel, mel_mask,
                    lambda x, y: F.mse_loss(x, y, reduction='mean')
                )
                
                # Compute duration losses
                duration_loss, relative_duration_loss = compute_duration_loss(
                    duration_pred.squeeze(-1),
                    batch['token_durations'],
                    attention_mask
                )
                
                # Combined loss
                mel_loss = MEL_LOSS_WEIGHT * l1_loss + MSE_LOSS_WEIGHT * mse_loss
                duration_term = DURATION_LOSS_WEIGHT * (duration_loss + 0.1 * relative_duration_loss)
                
                loss = mel_loss + duration_term
                
                # Clip loss to prevent explosion
                loss = torch.clamp(loss, max=100.0)
                
                # Check for NaN/Inf
                if not torch.isfinite(loss):
                    print(f"\nWarning: Non-finite loss detected: {loss.item()}")
                    continue
                
                # Scale loss more gradually during warmup
                if epoch < WARMUP_EPOCHS:
                    loss = loss * lr_scale
                
                # Gradient accumulation with scaling
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.adapter.parameters(), 
                        GRADIENT_CLIP * lr_scale
                    )
                    
                    if torch.isfinite(grad_norm):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.ema.update_parameters(self.adapter)
                    
                    self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                valid_batches += 1
                epoch_losses['l1_loss'] += l1_loss.item()
                epoch_losses['mse_loss'] += mse_loss.item()
                epoch_losses['duration_loss'] += duration_loss.item()
                epoch_losses['relative_duration_loss'] += relative_duration_loss.item()
                
                # Update progress bar
                progress.set_postfix({
                    'l1': f"{l1_loss.item():.4f}",
                    'mse': f"{mse_loss.item():.4f}",
                    'dur': f"{duration_loss.item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
                })
                
            except Exception as e:
                print(f"\nError in batch: {str(e)}")
                continue
        
        # Calculate averages
        for k in epoch_losses:
            epoch_losses[k] /= valid_batches
        
        print(f"\nEpoch {epoch} Component Averages:")
        for k, v in epoch_losses.items():
            print(f"Average {k}: {v:.4f}")
        
        return total_loss / valid_batches if valid_batches > 0 else float('inf')

    def combine_chunks(self, chunks, cross_fade_length=256):
        """Combine mel spectrogram chunks with cross-fading"""
        if len(chunks) == 1:
            return chunks[0]
        
        combined = chunks[0]
        for i in range(1, len(chunks)):
            # Cross-fade between chunks
            fade_out = torch.linspace(1, 0, cross_fade_length, device=DEVICE)
            fade_in = torch.linspace(0, 1, cross_fade_length, device=DEVICE)
            
            # Apply cross-fade
            overlap_start = combined.size(-1) - cross_fade_length
            overlap_end = combined.size(-1)
            
            combined[..., overlap_start:overlap_end] *= fade_out
            chunks[i][..., :cross_fade_length] *= fade_in
            
            # Combine chunks
            combined = torch.cat([
                combined[..., :-cross_fade_length],
                combined[..., -cross_fade_length:] + chunks[i][..., :cross_fade_length],
                chunks[i][..., cross_fade_length:]
            ], dim=-1)
        
        return combined

    def validate(self, epoch):
        self.ema.eval()
        valid_batches = len(self.val_dataloader)
        
        # Initialize loss tracking
        running_loss = 0.0
        component_losses = {
            'l1_loss': 0.0,
            'mse_loss': 0.0,
            'duration_loss': 0.0,
            'relative_duration_loss': 0.0
        }
        
        progress_bar = tqdm(self.val_dataloader, desc="Validating")
        
        with torch.no_grad():
            for batch in progress_bar:
                try:
                    # Create attention mask
                    attention_mask = torch.arange(batch['embeddings'].size(1), device=DEVICE)[None, :] < batch['emb_lengths'][:, None]
                    
                    # Forward pass with duration prediction
                    pred_mel, duration_pred = self.ema(
                        batch['embeddings'],
                        timesteps=torch.zeros(batch['embeddings'].size(0), device=DEVICE),
                        mask=attention_mask,
                        return_durations=True
                    )
                    
                    # Interpolate to match target length
                    pred_mel = F.interpolate(
                        pred_mel,
                        size=batch['mel_spec'].size(-1),
                        mode='linear',
                        align_corners=False
                    )
                    
                    # Create mel-level mask
                    mel_mask = torch.arange(batch['mel_spec'].size(-1), device=DEVICE)[None, :] < batch['mel_lengths'][:, None]
                    mel_mask = mel_mask.unsqueeze(1).float()
                    
                    # Normalize mels
                    pred_mel = normalize_mel(pred_mel)
                    target_mel = normalize_mel(batch['mel_spec'])
                    
                    # Compute losses
                    l1_loss = compute_masked_loss(
                        pred_mel, target_mel, mel_mask,
                        lambda x, y: F.l1_loss(x, y, reduction='mean')
                    )
                    
                    mse_loss = compute_masked_loss(
                        pred_mel, target_mel, mel_mask,
                        lambda x, y: F.mse_loss(x, y, reduction='mean')
                    )
                    
                    duration_loss, relative_duration_loss = compute_duration_loss(
                        duration_pred.squeeze(-1),
                        batch['token_durations'],
                        attention_mask
                    )
                    
                    # Combined loss calculation
                    mel_loss = MEL_LOSS_WEIGHT * l1_loss + MSE_LOSS_WEIGHT * mse_loss
                    duration_term = DURATION_LOSS_WEIGHT * (duration_loss + 0.1 * relative_duration_loss)
                    total_loss = mel_loss + duration_term
                    
                    # Update running totals
                    running_loss += total_loss.item()
                    component_losses['l1_loss'] += l1_loss.item()
                    component_losses['mse_loss'] += mse_loss.item()
                    component_losses['duration_loss'] += duration_loss.item()
                    component_losses['relative_duration_loss'] += relative_duration_loss.item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'val_l1': f"{l1_loss.item():.4f}",
                        'val_mse': f"{mse_loss.item():.4f}",
                        'val_dur': f"{duration_loss.item():.4f}"
                    })
                    
                except Exception as e:
                    print(f"\nError in validation batch: {str(e)}")
                    continue
        
        # Calculate averages
        if valid_batches > 0:
            running_loss /= valid_batches
            for k in component_losses:
                component_losses[k] /= valid_batches
            
            print("\nValidation Component Averages:")
            for k, v in component_losses.items():
                print(f"Average {k}: {v:.4f}")
        
        return running_loss

    def train(self):
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_path = CHECKPOINT_DIR / "adapter_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.adapter.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, best_path)
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                    break
            
            # Save checkpoint
            checkpoint_path = CHECKPOINT_DIR / f"adapter_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.adapter.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)

    def analyze_convergence(self, epoch):
        # Compare distributions
        f5_stats = self.get_f5_reference_stats()
        adapter_stats = self.get_current_stats()
        
        divergence = {
            'mel_range': abs(f5_stats['mel_range'] - adapter_stats['mel_range']),
            'duration_mean': abs(f5_stats['duration_mean'] - adapter_stats['duration_mean']),
            'f0_correlation': compute_f0_correlation(f5_stats['f0'], adapter_stats['f0'])
        }
        
        # Log convergence metrics
        self.training_history['epochs'][epoch]['convergence_metrics'] = {
            'f5_reference': f5_stats,
            'adapter_current': adapter_stats,
            'divergence': divergence
        }
        self._save_training_log()
        
        return divergence

    def compute_feature_matching_loss(self, pred_mel, target_mel):
        """Compute multi-scale feature matching loss"""
        losses = []
        
        # Multi-scale features
        scales = [2048, 1024, 512, 256]
        for scale in scales:
            if pred_mel.size(-1) >= scale:
                # Average pooling at different scales
                pred_features = F.avg_pool1d(pred_mel, kernel_size=scale//8, stride=scale//16)
                target_features = F.avg_pool1d(target_mel, kernel_size=scale//8, stride=scale//16)
                
                # Feature matching loss at this scale
                losses.append(F.l1_loss(pred_features, target_features))
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=pred_mel.device)

    def get_training_difficulty(self, epoch):
        """Gradually increase training difficulty"""
        if epoch < 5:
            # Start with simple cases
            return {
                'max_sequence_length': 50,
                'complexity_weight': 0.5
            }
        elif epoch < 10:
            # Increase complexity
            return {
                'max_sequence_length': 100,
                'complexity_weight': 0.8
            }
        else:
            # Full difficulty
            return {
                'max_sequence_length': None,
                'complexity_weight': 1.0
            }

    def get_f5_reference_stats(self):
        """Get statistics from F5-TTS reference outputs"""
        return {
            'mel_range': (-6.0, 6.0),  # Typical F5-TTS mel range
            'duration_mean': self.dataset.token_duration_stats['mean'],
            'f0': self.dataset.samples[0]['prosody']['f0_contour']  # Use first sample as reference
        }

    def get_current_stats(self):
        """Get current adapter output statistics"""
        with torch.no_grad():
            # Get a batch
            batch = next(iter(self.train_dataloader))
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Cap sequence lengths
            emb_len = min(batch['embeddings'].size(1), 512)
            mel_len = min(batch['mel_spec'].size(2), 512 * 8)
            
            # Truncate sequences
            batch['embeddings'] = batch['embeddings'][:, :emb_len]
            batch['mel_spec'] = batch['mel_spec'][:, :, :mel_len]
            batch['token_durations'] = batch['token_durations'][:, :emb_len]
            batch['emb_lens'] = torch.tensor([emb_len], device=DEVICE)
            batch['mel_lens'] = torch.tensor([mel_len], device=DEVICE)
            
            # Generate output
            pred_mel, duration_pred = self.adapter(
                batch['embeddings'],
                timesteps=torch.zeros(len(batch['embeddings']), device=DEVICE),
                mask=lens_to_mask(batch['emb_lens']),
                return_durations=True
            )
            
            return {
                'mel_range': (pred_mel.min().item(), pred_mel.max().item()),
                'duration_mean': duration_pred.mean().item(),
                'f0': batch['prosody']['f0'][:, :mel_len]  # Use truncated F0
            }

    def compute_spectral_loss(self, pred_mel, target_mel):
        """Compute loss in frequency domain"""
        pred_fft = torch.fft.rfft(pred_mel, dim=-1)
        target_fft = torch.fft.rfft(target_mel, dim=-1)
        
        # Magnitude loss
        mag_loss = F.l1_loss(pred_fft.abs(), target_fft.abs())
        
        # Phase loss (less weight since phase is less critical)
        phase_loss = F.l1_loss(pred_fft.angle(), target_fft.angle())
        
        return mag_loss + 0.1 * phase_loss

    def compute_consistency_loss(self, pred_mel, mel_mask):
        """Encourage temporal consistency"""
        # Temporal difference
        temp_diff = pred_mel[..., 1:] - pred_mel[..., :-1]
        temp_diff_smoothness = torch.mean(torch.abs(temp_diff))
        
        # Frequency difference
        freq_diff = pred_mel[:, 1:, :] - pred_mel[:, :-1, :]
        freq_diff_smoothness = torch.mean(torch.abs(freq_diff))
        
        return temp_diff_smoothness + freq_diff_smoothness

    def compute_temporal_consistency_loss(self, pred_mel, target_mel, mel_mask):
        """Multi-scale temporal consistency loss"""
        # First-order temporal difference
        pred_diff1 = pred_mel[..., 1:] - pred_mel[..., :-1]
        target_diff1 = target_mel[..., 1:] - target_mel[..., :-1]
        
        # Second-order temporal difference (acceleration)
        pred_diff2 = pred_diff1[..., 1:] - pred_diff1[..., :-1]
        target_diff2 = target_diff1[..., 1:] - target_diff1[..., :-1]
        
        # Adjust mask for differences
        mask_diff1 = mel_mask[..., 1:]
        mask_diff2 = mel_mask[..., 2:]
        
        # Compute losses at different scales
        velocity_loss = F.l1_loss(
            pred_diff1 * mask_diff1,
            target_diff1 * mask_diff1,
            reduction='mean'
        )
        
        acceleration_loss = F.l1_loss(
            pred_diff2 * mask_diff2,
            target_diff2 * mask_diff2,
            reduction='mean'
        )
        
        return velocity_loss + 0.5 * acceleration_loss

    def compute_prosody_loss(self, pred_mel, target_mel, mel_mask):
        """Match prosodic features like energy and rhythm"""
        # Energy matching
        pred_energy = torch.norm(pred_mel, dim=1, keepdim=True)
        target_energy = torch.norm(target_mel, dim=1, keepdim=True)
        
        energy_loss = F.l1_loss(
            pred_energy * mel_mask,
            target_energy * mel_mask,
            reduction='mean'
        )
        
        # Rhythm matching (using autocorrelation)
        pred_auto = F.conv1d(pred_mel, pred_mel[..., :32].flip(-1))
        target_auto = F.conv1d(target_mel, target_mel[..., :32].flip(-1))
        
        rhythm_loss = F.l1_loss(
            pred_auto * mel_mask[..., :pred_auto.size(-1)],
            target_auto * mel_mask[..., :target_auto.size(-1)],
            reduction='mean'
        )
        
        return energy_loss + 0.5 * rhythm_loss

    def get_progressive_schedule(self, epoch):
        """Progressive training schedule with smoother transitions and better balance"""
        base_schedule = {
            'mel_weight': 1.0,
            'spectral_weight': 0.3,
            'temporal_weight': 0.2,
            'prosody_weight': 0.2,
            'duration_weight': 0.3
        }
        
        if epoch < WARMUP_EPOCHS:
            # Focus on basic reconstruction with higher duration weight
            return {
                'mel_weight': 1.0,
                'spectral_weight': 0.2,
                'temporal_weight': 0.1,
                'prosody_weight': 0.1,
                'duration_weight': 0.5
            }
        elif epoch < 15:
            # Gradually increase complexity with better balance
            progress = (epoch - WARMUP_EPOCHS) / (15 - WARMUP_EPOCHS)
            return {
                'mel_weight': 1.0,
                'spectral_weight': 0.2 + 0.3 * progress,
                'temporal_weight': 0.1 + 0.3 * progress,
                'prosody_weight': 0.1 + 0.3 * progress,
                'duration_weight': 0.5 - 0.2 * progress
            }
        else:
            # Full complexity with balanced weights
            return {
                'mel_weight': 1.0,
                'spectral_weight': 0.5,
                'temporal_weight': 0.4,
                'prosody_weight': 0.4,
                'duration_weight': 0.3
            }

    def get_lr_scale(self, epoch, step):
        """Get learning rate scale based on warmup and decay"""
        # Warmup phase
        if epoch < WARMUP_EPOCHS:
            return min(1.0, (epoch * len(self.train_dataloader) + step + 1) / 
                      (WARMUP_EPOCHS * len(self.train_dataloader)))
        
        # Cosine decay after warmup
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

def compute_f0_correlation(f0_ref, f0_pred):
    """Compute correlation between reference and predicted F0 contours"""
    # Remove unvoiced regions (where F0=0)
    mask = (f0_ref > 0) & (f0_pred > 0)
    if not torch.any(mask):
        return 0.0
    
    f0_ref = f0_ref[mask]
    f0_pred = f0_pred[mask]
    
    # Compute Pearson correlation
    f0_ref = (f0_ref - f0_ref.mean()) / f0_ref.std()
    f0_pred = (f0_pred - f0_pred.mean()) / f0_pred.std()
    correlation = torch.mean(f0_ref * f0_pred)
    
    return correlation.item()

def duration_loss(pred_durations, target_durations, target_length=None):
    """Calculate duration loss with optional length matching"""
    loss = F.mse_loss(pred_durations, target_durations)
    
    if target_length is not None:
        # Add length matching term
        pred_total = pred_durations.sum(dim=1)
        target_total = target_durations.sum(dim=1)
        length_loss = F.mse_loss(pred_total, target_total)
        loss = loss + length_loss
        
    return loss

def main():
    from adapter.adapter import EnhancedEmbeddingAdapter
    
    # Initialize adapter
    adapter = EnhancedEmbeddingAdapter(
        llama_dim=3072,
        tts_dim=1024,
        depth=8,
        heads=8,
        dim_head=64,
        ff_mult=4
    )
    
    print(f"\nAdapter parameters: {sum(p.numel() for p in adapter.parameters())}")
    
    # Load vocoder once for training
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False)
    
    # Create trainer and start training, passing the existing adapter
    trainer = AdapterTrainer(adapter, "generated_dataset.pkl", vocoder=vocoder)
    
    # Add validation dataset size check
    if len(trainer.val_dataloader) == 0:
        print("Warning: Validation dataset is empty! Please check your data loading configuration.")
        print("Training will continue without validation.")
    
    print("\nStarting training...")
    trainer.train()

if __name__ == "__main__":
    main() 