import torch
import torch.nn as nn
import torch.optim as optim

from dataloader2.dataloader import create_data_loader
from models.DeepLabV3p import DeepLabV3Plus
from models.Unet import Unet
from dataloader2 import dataloader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
lr = 0.001
batch_size = 2
num_epochs = 10

# Define model

model = DeepLabV3Plus(n_classes=5)
model.to(device)


# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define data loaders
train_data_loader = create_data_loader('./dataset/', batch_size=batch_size, num_workers=0)

# Train model
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i, (images, annotations) in enumerate(train_data_loader):
        images = images.to(device)
        annotations = annotations.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(images)
        outputs = torch.argmax(outputs, dim=1)
        outputs = outputs.type(torch.long)
        loss = criterion(outputs, annotations.squeeze(1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_data_loader.dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train loss: {train_loss:.4f}')
