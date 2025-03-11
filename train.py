import torch
import torch.nn as nn
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
    
class MusicRNN(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, hidden_dim=128):
        super(MusicRNN, self).__init__()
        
        # If dimensions not provided, use defaults from feature_extraction
        if input_dim is None:
            input_dim = feature_extraction.get_feature_dim()
        if output_dim is None:
            output_dim = feature_extraction.get_num_classes()
            
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output
    
class Trainer:
    def __init__(self, dataset=None, model=None, midi_files=None, epochs=10, batch_size=32, learning_rate=0.001, device=None):
        # Create dataset from midi files if provided
        if dataset is None and midi_files is not None:
            self.dataset = MidiDataset(midi_files=midi_files)
            
            # Create model with proper dimensions if not provided
            if model is None:
                input_dim = feature_extraction.get_feature_dim()
                output_dim = feature_extraction.get_num_classes()
                self.model = MusicRNN(input_dim=input_dim, output_dim=output_dim)
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
        # Need to split the dataset into training, testing, and validation sets
        features = self.dataset.features
        labels = self.dataset.labels
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

        train_dataset = MidiDataset(X_train, y_train)
        test_dataset = MidiDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.custom_collate)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.custom_collate)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Track best model performance
        best_f1 = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            # Create progress bar for training loop
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for features, labels in progress_bar:
                # Move data to GPU if available
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass through RNN
                outputs = self.model(features)

                loss = criterion(outputs.transpose(1, 2), labels)
                optimizer.zero_grad()
                loss.backward()
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                
                # Update progress bar with current loss
                progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            
            # Evaluate the model at the end of each epoch
            print("Evaluating model...")
            f1 = self.evaluate(test_loader)
            avg_loss = total_loss / len(train_loader)
            
            # Display metrics with clear formatting
            print(f"Epoch {epoch + 1}/{self.epochs} Results:")
            print(f"  Training Loss: {avg_loss:.4f}")
            print(f"  Test Accuracy: {f1:.4f}")
            
            # Save the model only if it's the best so far
            if f1 > best_f1:
                best_f1 = f1
                
                # Create models directory if it doesn't exist
                models_dir = os.path.join(r"C:\Users\henry-cao-local\Desktop\Personal_Projects\Music_Generation\music_generation_midi\models")
                os.makedirs(models_dir, exist_ok=True)
                
                # Find the highest existing version number (only once, outside the loop)
                if not hasattr(self, 'model_version'):
                    self.model_version = 1
                    for file in os.listdir(models_dir):
                        if file.startswith("model_v") and file.endswith(".pt"):
                            try:
                                v = int(file.split("_v")[1].split(".")[0])
                                self.model_version = max(self.model_version, v + 1)
                            except:
                                pass
                
                # Create a unique model path with version number
                model_path = os.path.join(models_dir, f"model_v{self.model_version}.pt")
                
                # Save model state dictionary
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                }, model_path)
                
                print(f"✓ Saved new best model with note accuracy: {best_f1:.4f} to {model_path}")

    def evaluate(self, test_loader):
        self.model.eval()
        f1 = 0
        total_batches = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                _, predicted = torch.max(outputs, dim=2)
                
                # Calculate F1 score
                f1 += self.f1_score_metric(predicted, labels)
                total_batches += 1
        
        # Average F1 score across all batches
        avg_f1 = f1 / total_batches if total_batches > 0 else 0
        
        return avg_f1

    def f1_score_metric(self, predicted, labels):
        # Flatten predicted and labels
        predicted_flat = predicted.view(-1).cpu().numpy()
        labels_flat = labels.view(-1).cpu().numpy()
        
        # Calculate F1 score
        return f1_score(labels_flat, predicted_flat, average='macro')
