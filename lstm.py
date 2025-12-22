# send it to: victor.torres.c@uni.edu.pe

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class VLFEventDetector(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.3):
        super(VLFEventDetector, self).__init__()
        
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim//2, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=False)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim//2, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim//2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x, return_attention=False):
        # x shape: (batch_size, seq_len, input_dim)
        
        # LSTM layers
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # Attention mechanism
        attention_weights = self.attention(lstm2_out)  # (batch_size, seq_len, 1)
        context_vector    = torch.sum(attention_weights * lstm2_out, dim=1)  # (batch_size, hidden_dim)
        
        # Classification
        output = self.classifier(context_vector)
        
        if return_attention:
            return output, attention_weights.squeeze(-1)
        return output


def prepare_dataseqs(df, signal_column='NAA-filt', label_column='state', sequence_length=30):
    """Prepare sequences directly from DataFrame"""
    
    signal_data = df[signal_column].values.reshape(-1, 1)  #(n_samples, 1)
    labels = df[label_column].values  # (n_samples,)
    
    sequences = []
    sequence_labels = []
    
    for i in range(len(signal_data) - sequence_length):
        # Extract sequence
        seq_signal = signal_data[i:i+sequence_length]
        
        # Calculate gradient as additional feature
        gradient = np.gradient(seq_signal[:, 0])
        
        # Combine signal and gradient
        seq_combined = np.column_stack((seq_signal, gradient))
        
        sequences.append(seq_combined)
        
        # Use the label at the END of the sequence for prediction
        label = labels[i+sequence_length-1]
        sequence_labels.append(label)
    
    return np.array(sequences), np.array(sequence_labels)

def calculate_class_weights(labels):
    """Calculate class weights for imbalanced dataset"""
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # tuning params
    weights = {
        0: 1.0,    # Normal - baseline
        1: 60.0,   # Rise - HIGH priority (increase to catch more rises)
        2: 25.0    # Return - medium priority
    }
    
    print(f"Class distribution: {class_counts}")
    print(f"Using class weights: {weights}")
    
    return weights

def train_model(model, train_loader, val_loader, class_weights, num_epochs=100, patience=10):
    # wether cuda available, else 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Convert class weights to tensor
    weight_tensor = torch.FloatTensor([class_weights[i] for i in range(3)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    print(f" > train loss: ...", end="\r")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            print(f" > train loss: {train_loss}", end="\r")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, class_names=['Normal', 'Rise', 'Return']):
    """Evaluate model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("CONFUSION MATRIX:")
    print(cm)
    
    # F1 scores per class
    f1_scores = f1_score(all_labels, all_predictions, average=None)
    print("\nF1 SCORES:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {f1_scores[i]:.4f}")
    
    return all_predictions, all_labels, all_probabilities

def predict_single_sequence(model, sequence_data, device):
    """Make prediction on a single sequence"""
    model.eval()
    with torch.no_grad():
        sequence_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(device)
        output = model(sequence_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)
        
    return prediction.item(), probabilities.cpu().numpy()[0]


def test_weight_combinations(df, weight_combinations, signal_column='NAA-filt', label_column='state'):
    """Test different class weight combinations to find optimal balance"""
    
    sequences, sequence_labels = prepare_dataseqs(
        df, signal_column=signal_column, label_column=label_column
    )
    
    # Chronological split
    n_sequences = len(sequences)
    train_end = int(0.6 * n_sequences)
    val_end = int(0.8 * n_sequences)
    
    X_train = sequences[:train_end]
    y_train = sequence_labels[:train_end]
    X_val   = sequences[train_end:val_end]
    y_val   = sequence_labels[train_end:val_end]
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset   = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    best_f1 = 0
    best_weights = None
    best_model = None
    
    print("Testing weight combinations...")
    print("="*60)
    
    for i, weights in enumerate(weight_combinations):
        print(f"\nCombination {i+1}: {weights}")
        
        model = VLFEventDetector(input_dim=2, hidden_dim=64, num_layers=2, num_classes=3)
        model, _, _ = train_model(model, train_loader, val_loader, weights, num_epochs=50, patience=8)
        
        # Evaluate on validation set
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        val_predictions = []
        val_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        report = classification_report(val_true, val_predictions, output_dict=True)
        f1_rise = report['1']['f1-score']  # Focus on Rise class F1
        
        print(f"Rise F1-score: {f1_rise:.4f}")
        print(f"Rise Precision: {report['1']['precision']:.4f}")
        print(f"Rise Recall: {report['1']['recall']:.4f}")
        
        if f1_rise > best_f1:
            best_f1 = f1_rise
            best_weights = weights
            best_model = model
    
    print(f"\n{'='*60}")
    print(f"BEST WEIGHTS: {best_weights}")
    print(f"BEST RISE F1: {best_f1:.4f}")
    
    return best_model, best_weights

# 