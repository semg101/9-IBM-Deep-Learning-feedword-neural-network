#Practice: Use the Sequential Constructor to test Sigmoid, Tanh, and Relu Activations Functions on the MNIST Dataset
#Preparation
# Import the libraries we need for this lab

# Using the following line code to install the torchvision library
# !conda install -y torchvision
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np

#Neural Network Module and Training Function--------------------------------------------------------------------
#Define a function to train the model. In this case, the function returns a Python dictionary to store the training loss, and accuracy on the validation data
# Define the function to train the model
def train(model, criterion, train_loader, validation_loader, optimizer, epochs = 100):
    i = 0
    useful_stuff = {'training_loss': [],'validation_accuracy': []}  
    
    for epoch in range(epochs):
        for i,(x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())
        
        correct = 0
        for x, y in validation_loader:
            yhat = model(x.view(-1, 28 * 28))
            _, label = torch.max(yhat, 1)
            correct += (label == y).sum().item()
    
        accuracy = 100 * (correct / len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)
    
    return useful_stuff

#Make Some Data-------------------------------------------------------------------------------------------
#Load the training dataset by setting the parameter train to True and convert it to a tensor by placing a transform object in the argument transform:
# Create the training dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

#Load the testing dataset by setting the parameter train to False and convert it to a tensor by placing a transform object in the argument transform:
# Create a validation  dataset 
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Create the criterion function
criterion = nn.CrossEntropyLoss()

# Create the training data loader and validation data loader object
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

#Test Sigmoid, Tanh, and Relu and Train the Model-------------------------------------------------------------------
# Use the following parameters to construct the model
input_dim = 28 * 28
hidden_dim = 100
output_dim = 10

#try---------------------------------------------------------------------------
#Use nn.Sequential to build a one hidden layer neural network  modelwith a sigmoid activation to classify the 10 digits from the MNIST dataset.
# Practice: Use nn.sequential and Sigmoid function to create the model
learning_rate = 0.01

model=nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.Sigmoid(),
    nn.Linear(hidden_dim, output_dim),
)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=30)

#Try-----------------------------------------------------------------------------------------------------------------
#Use nn.Sequential to build a one hidden layer neural model_Tanh network with a Tanh activation to classify the 10 digits from the MNIST dataset.
# Practice: Use nn.sequential and Tanh function to create the model
learning_rate = 0.01

model_Tanh=nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, output_dim),
) 

optimizer = torch.optim.SGD(model_Tanh.parameters(), lr=learning_rate)
training_results_tanch = train(model_Tanh, criterion, train_loader, validation_loader, optimizer, epochs=30)

#Try--------------------------------------------------------------------------------------------------------------
#Use nn.Sequential to build a one hidden layer neural modelRelu network with a Relu activation to classify the 10 digits from the MNIST dataset.
# Practice: Use nn.sequential and Relu function to create the model
learning_rate = 0.01

modelRelu = torch.nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim),
)

optimizer = torch.optim.SGD(modelRelu.parameters(), lr=learning_rate)
training_results_relu = train(modelRelu, criterion, train_loader, validation_loader, optimizer, epochs=30)

#Analyze Results-----------------------------------------------------------------------------------------
#Compare the training loss for each activation:
# Compare the training loss
plt.plot(training_results_tanch['training_loss'], label='tanh')
plt.plot(training_results['training_loss'], label='sigmoid')
plt.plot(training_results_relu['training_loss'], label='relu')
plt.ylabel('loss')
plt.ylabel('loss')
plt.title('training loss iterations')
plt.legend()

#Compare the validation loss for each model: 
# Compare the validation loss
plt.plot(training_results_tanch['validation_accuracy'], label='tanh')
plt.plot(training_results['validation_accuracy'], label='sigmoid')
plt.plot(training_results_relu['validation_accuracy'], label='relu') 
plt.ylabel('validation accuracy')
plt.xlabel('epochs ')   
plt.legend()