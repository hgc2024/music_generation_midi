import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

class MidiDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
class MusicTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=6, num_heads=8, hidden_dim=512):
        super(MusicTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x  # Fixed missing return
    
class Trainer:
    def __init__(self, dataset, model, epochs=10, batch_size=32, learning_rate=0.001):
        self.dataset = dataset
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Move model to GPU if available
        self.model = model.to(self.device)

    def train(self):
        # Need to split the dataset into training, testing, and validation sets
        features = self.dataset.features
        labels = self.dataset.labels
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

        train_dataset = MidiDataset(X_train, y_train)
        test_dataset = MidiDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            self.model.train()
            for features, labels in train_loader:
                # Move data to GPU if available
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Evaluate the model
            self.evaluate(test_loader)
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    def evaluate(self, dataloader):
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for features, labels in dataloader:
                # Move data to GPU if available
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
            
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"Accuracy: {accuracy}, F1 Score: {f1}")
        return accuracy, f1