# Created by Rahul Kumar

# This code implements the Deep Neural Network for MNIST digit data classification

# Importing libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

# Function to train the network
def train_network(cnn,data_loader,optimizer,epoch):
    # Train the model
    cnn.train()
    losses = []
    counter = []
    correct = 0

    # Looping 1 epoch in batches
    for batch_idx, (data,target) in enumerate(data_loader):
        # Compute prediction error
        pred = cnn(data)
        loss = F.nll_loss(pred,target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == target).type(torch.float).sum().item()

        # Displaying the loss after each 100th batch
        if batch_idx%10 == 0:
            loss, cur_data = loss.item(), (batch_idx+1)*len(data)
            print(f'Train Loss : {loss:.6f}  images_trained: [{cur_data}/{len(data_loader.dataset)}]')
            losses.append(loss)
            counter.append(cur_data+(epoch*len(data_loader.dataset)))  
    correct /= len(data_loader.dataset)
    accuracy = correct*100.
    return losses,counter,accuracy

# Function to test the network
def test(cnn,data_loader):
    # Test the model
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
    accuracy = correct*100.
    print(f'Test set: Accuracy: {correct*100.:.2f}% Loss: {test_loss:.6f}')
    return losses,accuracy

# Function to plot the ground truth
def plotGroundTruth(train_data):
    figure = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()    
        plt.imshow(train_data.data[i], cmap='gray',interpolation='none')
        plt.title('Ground Truth:%i' %train_data.targets[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()
    figure

# Function to plot the loss of train data and test data
def plotLoss(train_counter,train_losses,test_counter,test_losses):
    figure = plt.figure()
    plt.plot(train_counter,train_losses,color='blue')
    plt.scatter(np.array(test_counter),np.array(test_losses),color='red')
    plt.legend(['Train loss','Test loss'])
    plt.xlabel('Number of training examples trained')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    figure

# Function to plot accuracy of train data and test data
def plotAccuracy(num_epoch,train_accuracy,test_accuracy):
    figure = plt.figure()
    epoch = np.arange(1,num_epoch+1)
    plt.scatter(epoch,train_accuracy,color='blue')
    plt.scatter(epoch,test_accuracy,color='red')
    plt.legend(['Train accuracy','Test accuracy'])
    plt.xlabel('Number of epoch trained')
    plt.ylabel('Accuracy(%)')
    plt.show()
    figure

# Defining main function
def main():
    # setting hyperparameters
    num_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 3
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False

    # Loading dataset
    train_data = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=torchvision.transforms.Compose([ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.0381,))]))
    test_data = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=torchvision.transforms.Compose([ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.0381,))]))
    train_loader = DataLoader(train_data,batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_data,batch_size=batch_size_test, shuffle=True)

    # PLotting ground truth
    plotGroundTruth(train_data=train_data)

    #initializing network
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    train_losses = []
    tr_losses = []
    train_counter = []
    tr_counter = []
    test_losses = []
    te_losses = []
    train_accuracy = []
    test_accuracy = []
    test_counter = [i*len(train_loader.dataset) for i in range(num_epochs+1)]

    # Training the network
    te_losses = test(cnn=network,data_loader=test_loader)
    test_losses.extend(te_losses)

    # Running network for epoch
    for i in range(num_epochs):
        print(f'-----------------------\nTraining Epoch: {i+1} \n-----------------------')
        tr_losses,tr_counter, tr_accuracy = train_network(cnn=network,data_loader=train_loader,optimizer=optimizer,epoch=i)
        train_losses.extend(tr_losses)
        train_counter.extend(tr_counter)
        train_accuracy.append(tr_accuracy)
        te_losses,te_accuracy = test(cnn=network,data_loader=test_loader)
        test_losses.extend(te_losses)
        test_accuracy.append(te_accuracy)

    plotLoss(train_counter=train_counter,train_losses=train_losses,test_counter=test_counter,test_losses=test_losses)
    
    plotAccuracy(num_epoch=num_epochs,train_accuracy=train_accuracy,test_accuracy=test_accuracy)

    torch.save(network.state_dict(),"CNNmodel.pth") 


if __name__ == "__main__":
    main()