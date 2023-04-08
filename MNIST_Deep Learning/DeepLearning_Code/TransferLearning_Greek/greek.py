# Created by Rahul Kumar

# This code implements the transfer learning for Greek letter classification
# using Pretrained network on MNIST data

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
import numpy as np
from CNN_network import MyNetwork

# Function to train the network
def train_network(cnn,data_loader,optimizer,epoch):
    #Train the model
    cnn.train()
    losses = []
    counter = []
    correct = 0

    # Looping 1 epoch in batches
    for batch_idx, (data,target) in enumerate(data_loader):      
        # compute prediction error
        pred = cnn(data)
        loss = F.nll_loss(pred,target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == target).type(torch.float).sum().item()

        # Displaying the loss after each 100th batch
        if batch_idx%5 == 0:
            loss, cur_data = loss.item(), (batch_idx+1)*len(data)
            print(f'Train Loss : {loss:.6f}  images_trained: [{cur_data}/{len(data_loader.dataset)}]')
            losses.append(loss)
            counter.append(cur_data+(epoch*len(data_loader.dataset)))  
    train_error = len(data_loader.dataset) - correct
    print(f"\nTrain error : {(train_error*100./len(data_loader.dataset)):.2f}%\n")
    return losses,counter,train_error

# Function to test the network
def test(cnn,data_loader):
    #Test the model
    cnn.eval()

    test_loss = 0
    correct = 0
    losses = []

    # Preventing gradient calculation
    with torch.no_grad():
        for data,target in data_loader:
            pred = cnn(data)
            test_loss += F.nll_loss(pred,target,size_average=False).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
    test_loss /= len(data_loader.dataset)
    losses.append(test_loss)
    correct /= len(data_loader.dataset)
    print(f'Test set: Accuracy: {correct*100.:.2f}% Loss: {test_loss:.6f}')
    return losses

# Function to plot the prediction
def plotGreekPrediction(model,data_loader):
    i=0
    fig = plt.figure()
    greek = {0:'alpha',1:'beta',2:'delta',3:'gamma',4:'lambda',5:'mu'}
    with torch.no_grad():
        for data,target in data_loader:
            output = model(data)
            i +=1
            image = np.array(data).reshape(28,28)
            if(i==10):
                break
            plt.subplot(3,3,i)
            plt.tight_layout()
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.title(f'\nPrediction: {greek[output.argmax(1).item()]}')        
    plt.show()
    fig

# Function to plot the training loss
def plotTrainLoss(train_counter, train_losses):
    figure = plt.figure()
    plt.plot(train_counter,train_losses,color='blue')
    plt.xlabel('Number of training examples trained')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    figure

# Function to plot the training error
def plotTrainError(iteration, error):
    figure = plt.figure()
    counter = np.arange(1,iteration+1)
    plt.plot(counter,error,color='red')
    #plt.legend(['Train Error'])
    plt.xlabel('Number of epoch')
    plt.ylabel('Training error(%)')
    plt.show()
    figure

# Defining class for Custom transformation of data
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28,28))
        x = torchvision.transforms.functional.invert(x)
        return x

# Defining main function
def main():
    network = MyNetwork()
    network.load_state_dict(torch.load("CNNmodel.pth"))
    
    #freezes the parameters for the whole network
    for param in network.parameters():
        param.requires_grad = False
    
    # Defining new layer and replacing with the exisitng last layer
    newfc2 = nn.Linear(50,6)
    network.fc2 = newfc2
    print(network)

    train_losses = []
    tr_losses = []
    train_counter = []
    tr_counter = []
    train_error = []

    # Setting the hyperparameter
    num_epochs=200
    learning_rate=0.01
    momentum=0.7
    random_seed = 3
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # Train Dataloader for the Greek dataset
    greek_train = DataLoader(
        torchvision.datasets.ImageFolder('/home/rahul/Desktop/Computer_Vision/assignment5/greek_train/greek_train',
        transform=torchvision.transforms.Compose([ToTensor(),GreekTransform(),torchvision.transforms.Normalize(
        (0.1307,),(0.3081,))])),batch_size=3,shuffle=True)
    
    # Test Dataloader for the Greek dataset
    greek_test = DataLoader(
        torchvision.datasets.ImageFolder('/home/rahul/Desktop/Computer_Vision/assignment5/greek_test',
        transform=torchvision.transforms.Compose([ToTensor(),GreekTransform(),torchvision.transforms.Normalize(
        (0.1307,),(0.3081,))])),batch_size=1,shuffle=True)
    
    for i,(image,label) in enumerate(greek_train):
        print(label)

    for i in range(num_epochs):
        print(f'-----------------------\nTraining Epoch: {i+1} \n-----------------------')
        tr_losses,tr_counter,tr_error = train_network(cnn=network,data_loader=greek_train,optimizer=optimizer,epoch=i)
        train_losses.extend(tr_losses)
        train_counter.extend(tr_counter)
        train_error.append(tr_error)

        #testing the network with each epoch
        test(cnn=network,data_loader=greek_test)
    
    # Funcitons to plot the training loss, training error and prediction
    plotTrainLoss(train_counter=train_counter,train_losses=train_losses)
    plotTrainError(iteration=num_epochs,error=train_error)
    plotGreekPrediction(model=network,data_loader=greek_test)


if __name__ == "__main__":
    main()