import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Utility function to check and set device
def get_device():
    """
    Automatically select the best available device (GPU if available, else CPU)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Convolutional Neural Network (CNN)
class CNNModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        """
        CNN for image classification tasks
        
        Args:
            input_channels (int): Number of input image channels (e.g., 1 for grayscale, 3 for RGB)
            num_classes (int): Number of output classes
        """
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward propagation through the network
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output predictions
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# 2. Recurrent Neural Network (RNN)
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        RNN for sequence classification
        
        Args:
            input_size (int): Size of each input vector
            hidden_size (int): Number of hidden units
            num_layers (int): Number of RNN layers
            num_classes (int): Number of output classes
        """
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN Layer
        self.rnn = nn.RNN(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward propagation through the RNN
        
        Args:
            x (torch.Tensor): Input sequence tensor
        
        Returns:
            torch.Tensor: Output predictions
        """
        # Initialize hidden state
        h0 = torch.zeros(
            self.num_layers, 
            x.size(0), 
            self.hidden_size
        ).to(x.device)
        
        # RNN forward prop
        out, _ = self.rnn(x, h0)
        
        # Take the last time step
        out = self.fc(out[:, -1, :])
        return out

# 3. Long Short-Term Memory (LSTM)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        LSTM for sequence classification
        
        Args:
            input_size (int): Size of each input vector
            hidden_size (int): Number of hidden units
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of output classes
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward propagation through the LSTM
        
        Args:
            x (torch.Tensor): Input sequence tensor
        
        Returns:
            torch.Tensor: Output predictions
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(
            self.num_layers, 
            x.size(0), 
            self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers, 
            x.size(0), 
            self.hidden_size
        ).to(x.device)
        
        # LSTM forward prop
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last time step
        out = self.fc(out[:, -1, :])
        return out

# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """
    Train the model
    
    Args:
        model (nn.Module): The neural network model
        train_loader (DataLoader): DataLoader for the training set
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run the training on
        num_epochs (int): Number of epochs to train
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation function
def evaluate_model(model, test_loader, device):
    """
    Evaluate the model
    
    Args:
        model (nn.Module): The neural network model
        test_loader (DataLoader): DataLoader for the test set
        device: Device to run the evaluation on
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# Example usage function
def main():
    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # CNN Example
    print("\nCNN Example:")
    batch_size, channels, height, width = 32, 1, 28, 28
    num_classes = 10
    
    # Create CNN model
    cnn_model = CNNModel(input_channels=channels, num_classes=num_classes).to(device)
    
    # Create random input and labels
    x_cnn = torch.randn(batch_size, channels, height, width)
    y_cnn = torch.randint(0, num_classes, (batch_size,))
    
    # Create DataLoader
    cnn_dataset = TensorDataset(x_cnn, y_cnn)
    cnn_loader = DataLoader(cnn_dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    
    # Train the CNN model
    train_model(cnn_model, cnn_loader, criterion, optimizer, device, num_epochs=5)
    
    # Evaluate the CNN model
    evaluate_model(cnn_model, cnn_loader, device)

    # RNN Example
    print("\nRNN Example:")
    sequence_length, batch_size, input_size = 10, 32, 20
    hidden_size, num_layers = 50, 2
    
    # Create RNN model
    rnn_model = RNNModel(
        input_size=input_size, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        num_classes=num_classes
    ).to(device)
    
    # Create random input sequence and labels
    x_rnn = torch.randn(batch_size, sequence_length, input_size)
    y_rnn = torch.randint(0, num_classes, (batch_size,))
    
    # Create DataLoader
    rnn_dataset = TensorDataset(x_rnn, y_rnn)
    rnn_loader = DataLoader(rnn_dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
    
    # Train the RNN model
    train_model(rnn_model, rnn_loader, criterion, optimizer, device, num_epochs=5)
    
    # Evaluate the RNN model
    evaluate_model(rnn_model, rnn_loader, device)

    # LSTM Example
    print("\nLSTM Example:")
    # Create LSTM model
    lstm_model = LSTMModel(
        input_size=input_size, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        num_classes=num_classes
    ).to(device)
    
    # Create random input sequence and labels
    x_lstm = torch.randn(batch_size, sequence_length, input_size)
    y_lstm = torch.randint(0, num_classes, (batch_size,))
    
    # Create DataLoader
    lstm_dataset = TensorDataset(x_lstm, y_lstm)
    lstm_loader = DataLoader(lstm_dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    
    # Train the LSTM model
    train_model(lstm_model, lstm_loader, criterion, optimizer, device, num_epochs=5)
    
    # Evaluate the LSTM model
    evaluate_model(lstm_model, lstm_loader, device)

if __name__ == "__main__":
    main()
