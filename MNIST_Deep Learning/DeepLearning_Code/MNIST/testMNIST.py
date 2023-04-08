# Created by Rahul Kumar

# This code implements testing of MNIST data using trained Deep Neural Network

# Importing libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

#class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)
    
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x

# Function to plot the prediciton of test data
def plotPrediction(model,data_loader,test_data):
    i=0
    fig = plt.figure()
    with torch.no_grad():
        for data,target in data_loader:
            output = model(data)
            i +=1
            print(f'Displaying Log softmax output for {i} test image:')
            for j in range(10):
                print(f"{output[0][j].item():.2f} " ,end =" ")
            print(f'\nPrediction: {output.argmax(1).item()}')
            print(f'Ground Truth: {target.item()}')
            if(i==10):
                break
            plt.subplot(3,3,i)
            plt.tight_layout()
            plt.imshow(test_data.data[i-1], cmap='gray')
            plt.axis('off')
            plt.title(f'\nPrediction: {output.argmax(1).item()}')        
    plt.show()
    fig

# Function to plot the prediction on new hand written digit
def plotNewPrediction(model,data_loader):
    i=0
    fig = plt.figure()
    with torch.no_grad():
        for data,target in data_loader:
            output = model(data)
            i +=1
            #output_arr = output.numpy()
            image = np.array(data).reshape(28,28)
            print(f'Displaying Log softmax output for {i} new Input image:')
            for j in range(10):
                print(f"{output[0][j].item():.2f} " ,end =" ")
            print(f'\nPrediction: {output.argmax(1).item()}')
            print(f'Ground Truth: {target.item()}')
            plt.subplot(4,3,i)
            plt.tight_layout()
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.title(f'\nPrediction: {output.argmax(1).item()}')        
    plt.show()
    fig

# Defining class for Custom transformation of data
class Custom_transform:
    def __init__(self):
        pass

    def __call__(self,x):
        x = transforms.functional.rgb_to_grayscale(x)
        x = transforms.functional.invert(x)
        return x

# Defining main function 
def main():
    #loading dataset
    test_data = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=torchvision.transforms.Compose([ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.0381,))]))
    test_loader = DataLoader(test_data,batch_size=1, shuffle=False)

    # Loading the trained network
    trained_network = MyNetwork()
    trained_network.load_state_dict(torch.load("CNNmodel.pth"))
    trained_network.eval()

    # Plotting prediction
    plotPrediction(model=trained_network,data_loader=test_loader,test_data=test_data)

    # Loading my handwritten dataset
    custom_test_loader = DataLoader(
        datasets.ImageFolder('/home/rahul/Desktop/Computer_Vision/assignment5/handwritten_digit',
        transform=transforms.Compose([ToTensor(),Custom_transform(),torchvision.transforms.Normalize(
        (0.1307,),(0.3081,))])),batch_size=1,shuffle=False)

    # Plotting prediction of handwritten digits
    plotNewPrediction(model=trained_network,data_loader=custom_test_loader)


if __name__ == "__main__":
    main()