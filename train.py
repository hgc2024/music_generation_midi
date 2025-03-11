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

class MidiDataset(Dataset):
    def __init__(self, features=None, labels=None, midi_files=None):
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
        # Debug output to inspect tensor shapes
        feature = torch.FloatTensor(self.features[idx])

        label = self.labels[idx]

        # Shift labels to valid range
        min_value = -208
        label = label - min_value

        # Convert to tensor
        label = torch.tensor(label, dtype=torch.long)
        label = torch.clamp(label, 0, 427)

        return feature, label
    
class MusicTransformer(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, num_layers=6, num_heads=8, hidden_dim=512, 
                 dropout=0.1, max_seq_length=2048):
        super(MusicTransformer, self).__init__()
        
        # Configure dimensions based on feature extraction module if not provided
        if input_dim is None:
            input_dim = feature_extraction.get_feature_dim()
        if output_dim is None:
            output_dim = feature_extraction.get_num_classes()
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Project input features to model dimension
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer blocks with relative attention
        encoder_layers = []
        for _ in range(num_layers):
            # Create custom relative attention transformer layer
            rel_attn_layer = RelativeAttentionTransformerLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dropout=dropout,
                max_seq_length=max_seq_length
            )
            encoder_layers.append(rel_attn_layer)
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc_intermediate = nn.Linear(hidden_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        # Project input to hidden dimension
        x = self.embedding(x)
        
        # Pass through transformer layers with relative attention
        for layer in self.transformer_encoder:
            x = layer(x)
        
        # Final processing
        x = self.layer_norm(x)
        x = self.fc_intermediate(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class RelativeAttentionTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, max_seq_length=2048):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Multi-head attention with relative position
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Relative position embeddings
        self.Er = nn.Parameter(torch.randn(max_seq_length * 2 - 1, self.head_dim))
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Apply layer normalization before attention (Pre-LN pattern)
        residual = x
        x = self.norm1(x)
        
        # Multi-head attention with relative position
        batch_size, seq_len, _ = x.size()
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Add relative positional bias
        relative_indices = torch.arange(seq_len, device=x.device).unsqueeze(1) - torch.arange(seq_len, device=x.device).unsqueeze(0)
        relative_indices += seq_len - 1  # Shift to [0, 2*seq_len-2]
        relative_pos_embeddings = self.Er[relative_indices]
        
        # Reshape to add to attention scores
        rel_pos_scores = torch.matmul(q.transpose(1, 2).reshape(-1, self.nhead, self.head_dim), 
                                      relative_pos_embeddings.transpose(-2, -1))
        rel_pos_scores = rel_pos_scores.view(batch_size, seq_len, self.nhead, seq_len)
        rel_pos_scores = rel_pos_scores.transpose(1, 2)
        
        # Add positional bias to content-based attention
        scores = scores + rel_pos_scores
        
        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection
        x = residual + self.dropout(attn_output)
        
        # Feed-forward network
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        
        return x
    
class Trainer:
    def __init__(self, dataset=None, model=None, midi_files=None, epochs=10, batch_size=32, learning_rate=0.001, device=None):
        # Create dataset from midi files if provided
        if dataset is None and midi_files is not None:
            self.dataset = MidiDataset(midi_files=midi_files)
            
            # Create model with proper dimensions if not provided
            if model is None:
                input_dim = feature_extraction.get_feature_dim()
                output_dim = feature_extraction.get_num_classes()
                self.model = MusicTransformer(input_dim=input_dim, output_dim=output_dim)
        else:
            self.dataset = dataset
            self.model = model
            
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Check if GPU is available
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Move model to GPU if available
        self.model = self.model.to(self.device)

    def custom_collate(self, batch):
        # Find the shortest sequence length in the batch
        min_length = min(item[0].shape[0] for item in batch)
        
        # Truncate all sequences to the minimum length
        features = [item[0][:min_length] for item in batch]
        labels = [item[1][:min_length] for item in batch]
        
        # Stack them into tensors
        features = torch.stack(features)
        labels = torch.stack(labels)
        
        return features, labels

    def train(self):
        # Split the dataset into training and test sets
        features = self.dataset.features
        labels = self.dataset.labels
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

        train_dataset = MidiDataset(X_train, y_train)
        test_dataset = MidiDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.custom_collate)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.custom_collate)

        # Set up optimizer with parameters from the paper (β1=0.9, β2=0.98, ε=10^-9)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0,  # Initial learning rate will be set by scheduler
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Implement learning rate scheduler from the paper
        # Custom learning rate schedule with warmup
        warmup_steps = 4000
        d_model = self.model.hidden_dim
        
        def lr_lambda(step):
            # Implementation of the formula from the Transformer paper
            step = max(1, step)  # Avoid division by zero
            return min(step**-0.5, step * (warmup_steps**-1.5)) * (d_model**-0.5)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Standard cross-entropy loss from the paper
        criterion = nn.CrossEntropyLoss()
        
        # Track best model performance
        best_composite_score = float('-inf')
        step = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            # Create progress bar for training loop
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for features, labels in progress_bar:
                # Move data to GPU if available
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass using the model's defined forward method
                outputs = self.model(features)

                # Calculate loss
                loss = criterion(outputs.transpose(1, 2), labels)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping as in the paper
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                step += 1
                
                total_loss += loss.item()
                
                # Update progress bar with current loss and learning rate
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    "batch_loss": f"{loss.item():.4f}", 
                    "lr": f"{current_lr:.6f}"
                })
            
            # Evaluate the model at the end of each epoch
            print("Evaluating model...")
            composite_score = self.evaluate(test_loader)
            avg_loss = total_loss / len(train_loader)
            
            # Display metrics with clear formatting
            print(f"Epoch {epoch + 1}/{self.epochs} Results:")
            print(f"  Training Loss: {avg_loss:.4f}")
            print(f"  Composite Score: {composite_score:.4f}")
            
            # Save the model only if it's the best so far
            if composite_score > best_composite_score:
                best_composite_score = composite_score
                
                # Create models directory if it doesn't exist
                models_dir = os.path.join(r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\music_generation_midi\models")
                os.makedirs(models_dir, exist_ok=True)
                
                # Check if we already determined a version number during initialization
                if not hasattr(self, 'model_path'):
                    # Find an unused version number for the model file
                    version = 5
                    while True:
                        model_path = os.path.join(models_dir, f"music_transformer_v{version}.pth")
                        if not os.path.exists(model_path):
                            break
                        version += 1
                    self.model_path = model_path
                
                # Save model state dictionary
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'best_score': best_composite_score
                }, self.model_path)
                
                print(f"✓ Saved new best model with composite_score: {best_composite_score:.4f} to {self.model_path}")

        # Print the final results
        print("Training complete!")
        print("Best composite score:", best_composite_score)
        print("Model saved to:", self.model_path)

    def evaluate(self, test_loader):
        self.model.eval()
        total_nll = 0
        total_samples = 0
        
        # Metrics from the paper: negative log-likelihood (NLL) is primary
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate negative log-likelihood loss 
                # (primary metric from the Music Transformer paper)
                criterion = nn.CrossEntropyLoss(reduction='sum')
                nll = criterion(outputs.transpose(1, 2), labels)
                
                total_nll += nll.item()
                total_samples += labels.numel()
                
                # For secondary analysis, calculate prediction accuracy
                _, predicted = torch.max(outputs, dim=2)
                note_accuracy = (predicted == labels).float().mean().item()
        
        # Normalize NLL by the number of tokens (as done in the paper)
        avg_nll = total_nll / total_samples
        perplexity = torch.exp(torch.tensor(avg_nll)).item()
        
        # Print metrics following the paper's evaluation methodology
        print(f"  Negative Log-Likelihood: {avg_nll:.4f}")
        print(f"  Perplexity: {perplexity:.4f}")
        print(f"  Note accuracy: {note_accuracy:.4f}")
        
        # Return negative perplexity as the composite score 
        # (lower perplexity is better, so we negate for optimization)
        return -perplexity
