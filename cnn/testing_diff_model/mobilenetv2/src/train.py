# type: ignore

import torch
import torch.nn as nn
from torch.optim import Adam
from data_loader import get_dataloaders
from model import get_model


def train_model(
        data_dir, num_classes=5, num_epochs=10, lr=0.001, batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes).to(device)
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)

    # Use BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Use Adam optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model.train()

        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Compute validation loss
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        print(f'Val Loss: {epoch_val_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'mobilenetv2_multilabel.pth')


if __name__ == "__main__":
    # Provide your dataset directory path
    data_dir = (
        '/Users/adityaravi/Desktop/project/'
        'AIDermaVision-MAJOR/implement/cnn/'
        'testing_diff_model/mobilenetv2/dataset'
    )
    train_model(data_dir)
