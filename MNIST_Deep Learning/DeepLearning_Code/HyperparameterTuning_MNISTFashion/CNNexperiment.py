# Created by Rahul Kumar

# This code implements the hyperparameter tuning

# Importing libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import random

#class definitions
class MyNetwork(nn.Module):
    def __init__(self,filter_channels,filter_size,dropout_rate):
        super(MyNetwork,self).__init__()
        self.conv_layer1 = nn.Conv2d(1,32,kernel_size=filter_size)
        self.conv_layer2 = nn.Conv2d(32,filter_channels,kernel_size=filter_size)
        self.conv_layer2_drop = nn.Dropout2d(p=dropout_rate)
        if filter_size == 3:
            self.fc1 = nn.Linear(5*5*filter_channels,80)
        else:
            self.fc1 = nn.Linear(4*4*filter_channels,80)
        self.fc2 = nn.Linear(80,10)
    
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv_layer1(x),2))
        x = F.relu(F.max_pool2d(self.conv_layer2_drop(self.conv_layer2(x)),2))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x

# Function to train the network
def train_network(model,train_loader,optimizer):
    model.train()

    for i,(image,label) in enumerate(train_loader):
        pred = model(image)
        loss = F.nll_loss(pred,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if i%100 == 0:
            loss, cur_data = loss.item(), (i+1)*len(image)

# Function to test the network
def test_network(model,test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for image,label in test_loader:
            pred = model(image)
            loss = F.nll_loss(pred,label)
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    accuracy = correct*100./len(test_loader.dataset)
    return accuracy

# Defining main function
def main():
    # declaring hyperparameter to be tuned
    learning_rate = [0.005,0.01,0.1]
    batch_size = [32,64,128]
    momentum = [0.5,0.7,0.9]
    filter_channels = [32,64,96]
    filter_size = [3,5]
    dropout_rate = [0.3,0.4,0.5]
    epoch = [3,5,8,10]

    # Defining the data transformation
    transform = transforms.Compose([ToTensor(),transforms.Normalize((0.5,),(0.5,))])

    # Loading the Fashion MNIST dataset
    train_data = datasets.FashionMNIST(root='data',download=True,train=True,transform=transform)
    test_data = datasets.FashionMNIST(root='data',download=True,train=False,transform=transform)

    best_accuracy = 0
    accuracy_list = []
    hyperparameter_list = []
    for i in range(100):
        # Implementing random search for different combinations
        lr_rate = random.choice(learning_rate)
        batch_sz = random.choice(batch_size)
        m = random.choice(momentum)
        filter_channel = random.choice(filter_channels)
        filter_sz = random.choice(filter_size)
        dropout = random.choice(dropout_rate)
        num_epoch = random.choice(epoch)

        # Printing the selected combination
        print(f'Start of experiment: {i+1}')
        print(f'Hyperparameters: {lr_rate},{batch_sz},{m},{filter_channel},{filter_sz},{dropout},{num_epoch}')

        # Creating model with selected parameters
        model = MyNetwork(filter_channels=filter_channel,filter_size=filter_sz,dropout_rate=dropout)
        optimizer = optim.SGD(model.parameters(), lr=lr_rate, momentum=m)

        # Creating dataloader for train and test set
        train_loader = DataLoader(train_data,batch_size=batch_sz,shuffle=True)
        test_loader = DataLoader(test_data,batch_size=1000,shuffle=False)
        accuracy = 0

        # Looping for epochs
        for e in range(num_epoch):
            train_network(model=model,train_loader=train_loader,optimizer=optimizer)
            accuracy = test_network(model=model,test_loader=test_loader)
            print(f'Accuracy: {accuracy}%')
        accuracy_list.append(accuracy)
        hyperparameter_list.append({'learning_rate': lr_rate, 'batch_size': batch_sz, 
                                    'momentum': m,'filter_channels': filter_channel, 
                                    'filter_size': filter_sz, 'dropout_rate': dropout, 'epoch': num_epoch})
        
        print(f'Hyperparameters: {lr_rate},{batch_sz},{m},{filter_channel},{filter_sz},{dropout},{num_epoch}')   
        print(f'End of experiment: {i+1}')
        
        # saving best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.state_dict()
            best_hyperparameters = {'learning_rate': lr_rate, 'batch_size': batch_sz, 
                                    'momentum': m,'filter_channels': filter_channel, 
                                    'filter_size': filter_sz, 'dropout_rate': dropout, 'epoch': num_epoch}
    
    # Printing best model
    print(f"Best hyperparameters: {best_hyperparameters}")
    print(f"Best accuracy: {best_accuracy}")
    print(accuracy_list)
    print(hyperparameter_list)

if __name__ == "__main__":
    main()

