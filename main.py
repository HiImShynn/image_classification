import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
from data.data import data_prep
from network.lenet import LeNet
from network.alexnet import AlexNet
from tqdm import tqdm

#define datasets parameters
image_size = 227

#define hyperparameters
batch_size = 64
lr = 1e-3
momentum = 0.8

# define datasets (choose an Dataset that comes along with Pytorch)
chosen_datasets = datasets.CIFAR10

# Setting up device
device = torch.device('cuda' if torch.cuda.is_available() else 'CPU')

# Train data, Test data 
train, test, img_dim = data_prep(
    chosen_datasets, batch_size= batch_size, shuffle= True, image_size= image_size
)
model = AlexNet(in_channels= img_dim, out_channels= 10).to(device= device)

# Setting up Optimizer and Loss function
optimizer = optim.SGD(model.parameters(), lr= lr, momentum= momentum)
criterion = nn.CrossEntropyLoss()


def train_model(num_epochs, train_data):
    model.train(True)
    running_loss = 0.0
    for epoch in range(num_epochs):
        loop = tqdm(
            enumerate(train_data), total= len(train_data), leave= True
        )
        for batch_index, (inputs, labels) in loop:
            # inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zeroing the grads so that they won't sum up
            optimizer.zero_grad()

            # feed data forward into the network
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Computing gradients in the backward direction
            loss.backward()

            # update weights based on computed gradients
            optimizer.step()

            # Sum up the loss over 1 batch
            running_loss += loss.item()
            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
            loop.set_postfix(loss = round(float(running_loss / batch_size), 4))
            loop.display()
            running_loss = 0.0
            # if (batch_index+1) % 50 == 49:
            #     # print(f'[{epoch + 1} -{batch_index + 1:5d}] loss {running_loss / 2000:.5f}')
            #     running_loss = 0.0
    print('Finished Training!')
    torch.save(model.state_dict(), './network/saved_models/AlexNet_MNIST.pth')
    print('Model has been saved at ./network/saved_models/AlexNet_MNIST.pth')

if __name__ == "__main__":
    print("Initializing... Training process will begin shortly!")
    train_model(10, train_data= train)
