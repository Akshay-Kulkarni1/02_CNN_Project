import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def main():
    #---------- Configuration ----------#
    BATCH_SIZE = 64
    DATA_DIR = '.'

    #---------- Transfrom: to tensor and Normalize to [-1, 1] ----------#
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #---------- Load the Dataset ----------#
    trainset = torchvision.datasets.CIFAR10(root = DATA_DIR, train = True, download = False, transform = transform)
    testset = torchvision.datasets.CIFAR10(root = DATA_DIR, train = False, download = False, transform = transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    #---------- Classes ----------#
    classes = trainset.classes

    #---------- Shape ----------#
    images, labels = next(iter(trainloader))
    print(images.shape)

    #---------- Shape ----------#
    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # Show a batch of images with labels
    imshow(torchvision.utils.make_grid(images[:8]))
    print(' '.join(f'{classes[labels[j]]}' for j in range(8)))

if __name__ == '__main__':
    main()