import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets

from data.data import data_prep
from network.lenet import LeNet

import time

#setting up device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#define hyperparameters
batch_size = 64
learning_rate = 1e-3
momentum = 0.8

#getting data to be loaded
chosen_dataset = datasets.FashionMNIST
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
        
        begin = time.time()
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
        
        end = time.time()
        epoch_acc = round(total_correct / (len(train_data)*batch_size), 3)
        epoch_loss = round(total_loss / len(train_data), 3)
        elapse_time = round(end - begin, 3)
        print("Epoch {}/{} --- Accuracy: {} --- Loss: {} --- Time: {}".format(
            epoch + 1, num_epochs, epoch_acc, epoch_loss, elapse_time
        ))
    
    path = "./network/saved_models/FashionMNIST_LeNet.pth"
    torch.save(model.state_dict(), path)
    print("Model saved!")

if __name__ == "__main__":
    train(50, train_data)