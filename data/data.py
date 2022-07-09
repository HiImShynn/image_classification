import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

def get_mean_std(data_loader):
    # VAR[x] = E[x**2] - E[x]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    # Data only, doesn't need the targets (labels)
    for data, _ in data_loader:
        # Sum of Number_of_instances, Instance_heights, Instance_width
        channels_sum += torch.mean(data, dim= [0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim= [0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std  = (channels_squared_sum / num_batches - mean**2)**0.5

    print(f'Mean of this datasets: {str(mean)}')
    print(f'Standard Deviation of this datasets: {str(std)}')

    return mean, std

def data_prep(chosen_dataset, batch_size= 64, shuffle= False, image_size = int):
    
    dummy_train = DataLoader(
        chosen_dataset(
            './data/datasets', train= True, 
            transform= transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ]), download= True
        ),
        batch_size= batch_size,
        shuffle= False,
    )
    
    data_mean, data_std= get_mean_std(dummy_train)

    trans = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean= data_mean, std= data_std)
    ])

    train_data = DataLoader(
        chosen_dataset(
            './data/datasets', train= True, 
            transform= trans, download= True
        ),
        batch_size= batch_size,
        shuffle= shuffle,
    )

    test_data = DataLoader(
        chosen_dataset(
            './data/datasets', train= False,
            transform= trans, download= True
        ),
        batch_size= batch_size,
        shuffle= shuffle
    )

    #print out numbers of instances has been loaded into train_data and test_data
    print('train_data (has {} items) has been loaded successfully !'.format(
        int(len(train_data)) * batch_size)
    )

    print('test_data (has {} items) has been loaded successfully !'.format(
        int(len(test_data)) * batch_size)
    )

    # Return train_loader, test_loader (data, labels) and amount of dimensions.
    return train_data, test_data, len(data_mean)

if __name__ == "__main__":
    from torchvision import datasets
    batch_size = 10
    chosen_dataset = datasets.CIFAR10
    train, test, img_dim = data_prep(
        chosen_dataset= chosen_dataset, batch_size= batch_size, shuffle= True, image_size= 28
    )
    
    print(img_dim)