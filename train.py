from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from data.data import data_prep
from network.lenet import LeNet

#setting up device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#define hyperparameters
batch_size = 64
learning_rate = 1e-3
momentum = 0.9

#getting data to be loaded
chosen_dataset = datasets.MNIST
train_data, test_data, img_dim = data_prep(
    chosen_dataset= chosen_dataset, batch_size= batch_size, shuffle= True, image_size= 32
)

#defining our model
model = LeNet(img_dim, 10).to(device)
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

def get_num_correct(outputs, targets):
    return outputs.argmax(dim= 1).eq(targets).sum().item()

def train(num_epochs, train_data):
    for epoch in range(num_epochs):

        total_loss = 0
        total_correct = 0

        for batch in train_data:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += get_num_correct(outputs, labels)
        
        print("Epoch:", epoch + 1, "total_correct:", total_correct, "loss:", total_loss)
    print(total_correct / len(train_data))

if __name__ == "__main__":
    train(10, train_data)