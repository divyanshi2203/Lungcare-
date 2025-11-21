import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# Define dataset transforms
image_size = 64  # target size for images
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # resize to 64x64
    transforms.ToTensor(),                       # convert to tensor (channels x H x W)
    transforms.Normalize(mean=[0.5, 0.5, 0.5],   # normalize to [-1,1] range (approx)
                         std=[0.5, 0.5, 0.5])
])

# Load training dataset
train_data = torchvision.datasets.ImageFolder(root='data/train', transform=transform)
# (Optional) Load validation dataset, if available
val_data = torchvision.datasets.ImageFolder(root='data/val', transform=transform)

print("Classes detected:", train_data.class_to_idx)  # e.g., {'cancer': 0, 'normal': 1}

from torch.utils.data import DataLoader

batch_size = 32  # you can adjust this based on memory
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)



class CancerNet(nn.Module):
    def __init__(self):
        super(CancerNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)   # conv layer 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)  # conv layer 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                      # pool reduces size by 2

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)  # conv layer 3
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)  # conv layer 4
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After two pooling operations, input image 64x64 becomes 13x13 with 64 channels (calculation shown below).
        self.fc1 = nn.Linear(64 * 13 * 13, 128)  # 64*13*13 = 10816, number of features after flattening
        self.fc2 = nn.Linear(128, 2)            # 2 output classes: [0] cancer, [1] normal (or vice versa)
    
    def forward(self, x):
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)            # after this, spatial size is halved
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)            # second pooling

        x = x.view(x.size(0), -1)    # flatten the tensor for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)              # no activation here; we'll apply softmax or interpret logits in loss
        return x


# Initialize model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CancerNet().to(device)                          # move model to GPU if available:contentReference[oaicite:20]{index=20}
criterion = nn.CrossEntropyLoss()                       # loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)    # Adam optimizer (could also use SGD)

# Training loop
num_epochs = 10  # you can adjust number of epochs
for epoch in range(1, num_epochs+1):
    model.train()   # set model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)   # move data to device:contentReference[oaicite:21]{index=21}

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)          

        # Backpropagation and optimization
        optimizer.zero_grad()   # zero out previous gradients:contentReference[oaicite:22]{index=22}
        loss.backward()         # compute gradients for this batch
        optimizer.step()        # update parameters

        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # (Optional) Validate on val_data to check accuracy:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # no need to compute gradients during validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)   # get index of max logit = predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total if total > 0 else 0
    print(f"Validation Accuracy: {val_acc:.2f}%")
# Save the trained model weights

torch.save(model.state_dict(), "cancer_detector.pth")
print("Model saved to cancer_detector.pth")
