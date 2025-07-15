import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# ---------- Helper to show images ----------
def imshow(img):
    img = img / 2 + 0.5  # unnormalize from [-1,1] to [0,1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

# ---------- Define CNN model ----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------- Main function ----------
def main():
    # ---------- Config ----------
    BATCH_SIZE = 64
    DATA_DIR = '.'

    # ---------- Transforms ----------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    # ---------- Load datasets ----------
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = trainset.classes

    # ---------- Visualize some data ----------
    images, labels = next(iter(trainloader))
    print("Sample batch shape:", images.shape)
    imshow(torchvision.utils.make_grid(images[:8]))
    print(' '.join(f'{classes[labels[j]]}' for j in range(8)))

    # ---------- Setup device and model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    net = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # ---------- Training loop ----------
    for epoch in range(5):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch+1}, {i+1:5d}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
    print("Finished Training")

    # ---------- Quick evaluation ----------
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print("Predicted:", ' '.join(f'{classes[predicted[j]]}' for j in range(8)))
    print("Actual:   ", ' '.join(f'{classes[labels[j]]}' for j in range(8)))
    imshow(torchvision.utils.make_grid(images[:8].cpu()))

    # ---------- Evaluate on the whole test set ----------
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the 10000 test images: {100 * correct / total:.2f}%")

# ---------- Windows-safe launch ----------
if __name__ == "__main__":
    main()
