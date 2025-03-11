import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import feature_extraction
import numpy as np
from tqdm import tqdm
import os
import math

class MidiDataset(Dataset):
    def __init__(self, features=None, labels=None, midi_files=None, piano_rolls=None):
        if features is not None and labels is not None:
            self.features = features
            self.labels = labels
        elif midi_files is not None:
            # Extract features using feature_extraction module
            self.features = []
            self.labels = []
            for file_path, label in midi_files:
                feature = feature_extraction.extract_features(file_path)
                self.features.append(feature)
                self.labels.append(label)
            self.features = np.array(self.features)
            self.labels = np.array(self.labels)
        else:
            raise ValueError("Either provide features and labels or midi_files")
            
        # Ensure all features have the same length by padding or truncating
        if len(self.features) > 0:
            # Find the max length among all features
            max_len = max(len(f) for f in self.features)
            # Pad or truncate all features to the same length
            for i in range(len(self.features)):
                if len(self.features[i]) < max_len:
                    # Pad with zeros
                    self.features[i] = np.pad(self.features[i], (0, max_len - len(self.features[i])))

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Return features and labels for a specific index.
        Ensures all sequences have consistent length.
        """
        # Define target sequence length based on piano roll (100 timesteps)
        seq_len = 100
        
        # Get piano roll and ensure it has consistent length
        piano_roll = self.features[idx]
        if len(piano_roll) > seq_len:
            piano_roll = piano_roll[:seq_len]
        elif len(piano_roll) < seq_len:
            # Pad with zeros if shorter
            pad_width = ((0, seq_len - len(piano_roll)), (0, 0))
            piano_roll = np.pad(piano_roll, pad_width)
        
        # Process durations if available
        duration = None
        if hasattr(self, 'durations') and len(self.durations) > idx:
            duration = self.durations[idx]
            if len(duration) > seq_len:
                duration = duration[:seq_len]
            elif len(duration) < seq_len:
                # Pad with zeros if shorter
                duration = np.pad(duration, (0, seq_len - len(duration)))
        
        # Process chords if available
        chord = None
        if hasattr(self, 'chords') and len(self.chords) > idx:
            chord = self.chords[idx]
            if len(chord) > seq_len:
                chord = chord[:seq_len, :]
            elif len(chord) < seq_len:
                # Pad with zeros if shorter
                pad_width = ((0, seq_len - len(chord)), (0, 0))
                chord = np.pad(chord, pad_width)
        
        # Create features tensor
        features = torch.FloatTensor(piano_roll)  # Shape: [seq_len, 128]
        
        # Add duration and chord features if available
        if duration is not None:
            duration_tensor = torch.FloatTensor(duration).unsqueeze(-1)  # [seq_len, 1]
            features = torch.cat([features, duration_tensor], dim=1)     # [seq_len, 129]
        
        if chord is not None:
            chord_tensor = torch.FloatTensor(chord)  # [seq_len, 4]
            features = torch.cat([features, chord_tensor], dim=1)  # [seq_len, 133] or [seq_len, 132]
        
        # Create labels (assuming next timestep prediction or shifted sequence)
        # Option 1: Use piano roll as labels (for reconstruction)
        labels = torch.LongTensor(piano_roll)
        
        # Option 2: Shift by 1 for next-step prediction
        # labels = torch.FloatTensor(piano_roll[1:, :])
        # features = torch.FloatTensor(piano_roll[:-1, :])
        
        # Option 3: If you have specific target values
        # labels = torch.LongTensor(self.targets[idx])
        
        # Ensure labels are in valid range for classification (if using CrossEntropyLoss)
        if labels.min() < 0:
            # Shift negative values to be non-negative
            offset = abs(labels.min().item())
            labels = labels + offset
        
        # Clamp to ensure within output_dim range (assuming 428 as in your model)
        labels = torch.clamp(labels, 0, 427)
        
        return features, labels
    
class MusicTransformer(nn.Module):
    def __init__(self, input_dim=128, output_dim=128, 
                 d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, 
                 max_seq_length=100, use_rel_pos=True):
        """
        Initialize the Music Transformer model.
        
        Args:
            input_dim: Dimension of the input features (usually 128 for MIDI)
            output_dim: Dimension of the output (usually same as input for music generation)
            d_model: Dimension of the model's hidden layers
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network in transformer
            dropout: Dropout rate
            max_seq_length: Maximum sequence length for positional encodings
            use_rel_pos: Whether to use relative positional encodings
        """
        super(MusicTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.use_rel_pos = use_rel_pos
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Create transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_rel_pos=use_rel_pos,
                max_seq_length=max_seq_length
            ) for _ in range(num_layers)
        ])
        
        # Final output projection
        self.output = nn.Linear(d_model, output_dim)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # Input embedding - ensure proper dimensions
        batch_size, seq_len, _ = x.shape
        
        # Check sequence length
        if seq_len > self.max_seq_length:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum {self.max_seq_length}")
        
        # Apply embedding
        x = self.embedding(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Output projection
        output = self.output(x)
        
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 use_rel_pos=True, max_seq_length=100):
        super(TransformerEncoderLayer, self).__init__()
        
        # Self-attention with relative positional encoding
        self.self_attn = MultiHeadAttentionWithRelPos(
            d_model, nhead, dropout=dropout,
            use_rel_pos=use_rel_pos,
            max_seq_length=max_seq_length
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization and dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Self-attention block
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward block
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x
    
class MultiHeadAttentionWithRelPos(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, 
                 use_rel_pos=True, max_seq_length=100):
        super(MultiHeadAttentionWithRelPos, self).__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_rel_pos = use_rel_pos
        self.max_seq_length = max_seq_length
        
        # Projection layers
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize relative positional encoding
        if use_rel_pos:
            self.rel_pos_enc = self._generate_relative_positions(max_seq_length)
    
    def _generate_relative_positions(self, length):
        # Create relative positional encodings
        # This can be a learned table or fixed sinusoidal encoding
        pos_enc = torch.zeros(length, self.d_model)
        position = torch.arange(0, length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * 
            -(math.log(10000.0) / self.d_model)
        )
        
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer so it's saved with the model
        self.register_buffer('rel_pos_encoding', pos_enc)
        return pos_enc

    def forward(self, x):
        """
        Forward pass for multi-head attention with relative positional encoding.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Linear projections for query, key, value
        q = self.query(x)  # [batch_size, seq_len, d_model]
        k = self.key(x)    # [batch_size, seq_len, d_model]
        v = self.value(x)  # [batch_size, seq_len, d_model]
        
        # 2. Reshape to separate heads
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim)  # [batch, seq, head, dim]
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim)  # [batch, seq, head, dim]
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim)  # [batch, seq, head, dim]
        
        # 3. Transpose for attention calculation
        q = q.transpose(1, 2)  # [batch, head, seq, dim]
        k = k.transpose(1, 2)  # [batch, head, seq, dim]
        v = v.transpose(1, 2)  # [batch, head, seq, dim]
        
        # 4. Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, head, seq, seq]
        
        # 5. Add relative positional encoding (if enabled)
        if self.use_rel_pos:
            # Ensure we don't exceed max_seq_length
            rel_pos = self.rel_pos_encoding[:seq_len, :self.d_model]
            
            # Project positional encoding to key space
            rel_pos_k = rel_pos.view(seq_len, self.nhead, self.head_dim)  # [seq, head, dim]
            rel_pos_k = rel_pos_k.transpose(0, 1)  # [head, seq, dim]
            
            # For each head and position i, compute attention scores with all other positions
            rel_scores = torch.zeros(batch_size, self.nhead, seq_len, seq_len, device=x.device)
            
            for b in range(batch_size):
                for h in range(self.nhead):
                    # Calculate relative position scores efficiently
                    q_b_h = q[b, h]  # [seq, dim]
                    pos_k_h = rel_pos_k[h]  # [seq, dim]
                    
                    # Calculate scores between each query position and all possible relative positions
                    for i in range(seq_len):
                        # This computes the attention score between position i and all other positions
                        for j in range(seq_len):
                            # Relative distance
                            rel_dist = i - j + seq_len - 1  # Shift to ensure positive index
                            if rel_dist < seq_len:
                                rel_scores[b, h, i, j] = torch.dot(q_b_h[i], pos_k_h[rel_dist % seq_len])
            
            # Add relative position scores to attention scores
            scores = scores + rel_scores
        
        # 6. Scale attention scores
        scores = scores / math.sqrt(self.head_dim)
        
        # 7. Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch, head, seq, seq]
        attn_weights = self.dropout(attn_weights)
        
        # 8. Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, head, seq, dim]
        
        # 9. Transpose and reshape attention output
        attn_output = attn_output.transpose(1, 2)  # [batch, seq, head, dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)  # [batch, seq, d_model]
        
        # 10. Apply output projection
        output = self.out_proj(attn_output)  # [batch, seq, d_model]
        
        return output


class Trainer:
    def __init__(self, dataset, model, epochs=10, batch_size=32, 
                 learning_rate=0.001, device='cuda', 
                 val_split=0.1, seed=42):
        """
        Initialize the trainer.
        
        Args:
            dataset: The MidiDataset instance
            model: The MusicTransformer model
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cuda' or 'cpu')
            val_split: Validation split ratio
            seed: Random seed for reproducibility
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        print(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(seed)
        
        self.model = model.to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Split dataset into train and validation
        dataset_size = len(dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Custom collate function to handle variable length sequences
        def collate_fn(batch):
            # Extract features and labels from batch
            features = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            
            # Get sequence length (should be consistent due to our dataset __getitem__)
            seq_len = features[0].shape[0]
            
            # Stack into tensors
            features = torch.stack(features)
            labels = torch.stack(labels)
            
            return features, labels
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            pin_memory=(self.device == 'cuda')
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            pin_memory=(self.device == 'cuda')
        )
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding tokens if needed
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
    
    def train(self):
        """Train the model for specified number of epochs."""
        for epoch in range(1, self.epochs + 1):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            total_samples = 0
            
            # Create a progress bar for training batches
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
            
            for features, labels in progress_bar:
                # Move data to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Ensure labels are in correct format and range
                labels = labels.long()  # Convert to LongTensor for CrossEntropyLoss
                
                # If using piano rolls as inputs and trying to predict next timestep:
                # labels should be within output_dim range
                if labels.max() >= self.model.output_dim:
                    labels = torch.clamp(labels, 0, self.model.output_dim - 1)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                try:
                    outputs = self.model(features)
                    
                    # Prepare outputs and labels for loss calculation
                    # For CrossEntropyLoss with sequence data, reshape:
                    batch_size, seq_len, output_dim = outputs.shape
                    outputs = outputs.view(-1, output_dim)  # [batch*seq_len, output_dim]
                    labels = labels.view(-1)  # [batch*seq_len]
                    
                    # Calculate loss
                    loss = self.criterion(outputs, labels)
                    
                    # Check loss validity
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Invalid loss value: {loss.item()}")
                        continue
                        
                    # Backward pass and optimize
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total = labels.size(0)
                    correct = (predicted == labels).sum().item()
                    
                    # Update metrics
                    batch_loss = loss.item()
                    train_loss += batch_loss * labels.size(0)
                    train_acc += correct
                    total_samples += total
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'batch_loss': f"{batch_loss:.4f}", 
                        'avg_loss': f"{train_loss/total_samples:.4f}",
                        'accuracy': f"{train_acc/total_samples:.4f}"
                    })
                    
                except RuntimeError as e:
                    print(f"Error in batch: {e}")
                    if "CUDA out of memory" in str(e):
                        print("CUDA out of memory. Trying to recover...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            # Calculate average training metrics
            train_loss /= total_samples
            train_acc /= total_samples
            self.train_losses.append(train_loss)
            
            # Evaluation phase
            val_loss, val_acc = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch}/{self.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def evaluate(self, data_loader):
        """Evaluate the model on a dataset."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, labels in data_loader:
                # Move data to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Ensure labels are in correct format and range
                labels = labels.long()
                if labels.max() >= self.model.output_dim:
                    labels = torch.clamp(labels, 0, self.model.output_dim - 1)
                
                try:
                    # Forward pass
                    outputs = self.model(features)
                    
                    # Reshape for loss calculation
                    batch_size, seq_len, output_dim = outputs.shape
                    outputs = outputs.view(-1, output_dim)
                    labels = labels.view(-1)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, labels)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total = labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Update metrics
                    val_loss += loss.item() * total
                    total_samples += total
                
                except RuntimeError as e:
                    print(f"Error during evaluation: {e}")
                    continue
        
        # Calculate average validation metrics
        val_loss /= total_samples if total_samples > 0 else 1
        val_acc = correct / total_samples if total_samples > 0 else 0
        
        return val_loss, val_acc
    
    def save_model(self, path):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"Model loaded from {path}")