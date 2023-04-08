# Created by Rahul Kumar

# This code implements the visualization of filters and its effect on the input image

# Importing libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from CNN_network import MyNetwork

# Function to plot weights/filters of first layer
def plotWeights(weights):
    fig = plt.figure()
    for i in range(10):
        print(f'filter {i} weights:')
        print(f'{weights.data[i][0].numpy()}')
        print(f'filter {i} shape: {weights.data[i][0].numpy().shape}\n')
        plt.subplot(3,4,i+1)
        plt.imshow(weights.data[i][0])
        plt.title(f'filter {i}')
        plt.axis('off')
    plt.show()
    
# Function to plot the filtered image
def plotFilterImage(weights,image):
    fig = plt.figure()
    with torch.no_grad():
        for i in range(10):
            Filter = weights.data[i][0].numpy()
            filterImage = cv2.filter2D(image,-1,Filter)
            plt.subplot(5,4,2*i+1)
            plt.imshow(weights.data[i][0], cmap='gray')
            plt.axis('off')
            plt.subplot(5,4,2*i+2)
            plt.imshow(filterImage, cmap='gray')
            plt.axis('off')
        plt.show()
        fig

def main():
    # Loading the trained network    
    trained_network = MyNetwork()
    trained_network.load_state_dict(torch.load("CNNmodel.pth"))

    # Loading the training data
    train_data = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=torchvision.transforms.Compose([ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.0381,))]))
    image = train_data.data[0]
    image_arr = image.numpy()
    weights = trained_network.conv1.weight # extracting the weights

    # weights visualisation
    plotWeights(weights=weights)

    # Filtered image visualisation
    plotFilterImage(weights=weights,image=image_arr)


if __name__ == "__main__":
    main()