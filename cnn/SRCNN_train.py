import torch
from torch.utils.data import DataLoader, Subset
from torch import narrow
from src.datasets import *
from model import *
from multiprocessing import Manager
from src.datasets import load_dataset_JIF
import os
from ipywidgets import interact_manual, Layout
import ipywidgets as widgets
from glob import glob
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from src.datasets import load_dataset_JIF, SatelliteDataset, DictDataset, make_transforms_JIF
from src.datasources import S2_ALL_12BANDS
from multiprocessing import Manager
from src.datasources import *
import warnings
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
from src.plot import showtensor
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from torchvision.utils import save_image

# Parameters for loading the datasets. 
BATCH_SIZE = 8
HOLDOUT = 0.2
PATCHES = 256
FRAMES = 8
WIDTH, HEIGHT = 400, 400


class CustomImageDataset():
    def __init__(self, feature, target):
        self.feature = feature
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        lr_image = self.feature[idx]
        hr_target = self.target[idx]
        return lr_image, hr_target

def dataloaders():    
    multiprocessing_manager = Manager()
    list_of_aois = list(
    pd.read_csv("../../worldstrat/pretrained_model/stratified_train_val_test_split.csv", index_col=1).index
    )
    kws = dict(
    input_size=(400, 400),
    output_size=(1024, 1024),
    normalize_lr=False,
    root="../../../../scratch/sun.jiam/dataset/",
    radiometry_depth=12,
    lr_bands_to_read="true_color",
    chip_size=(400,400),
    )
    dataloaders = load_dataset_JIF(**kws)
    train = dataloaders["train"].dataset
    val = dataloaders["val"].dataset
    test = dataloaders["test"].dataset
    train = Subset(dataloaders["train"].dataset, list(range(0, 33)))
    test = Subset(dataloaders["test"].dataset, list(range(0, 9)))
    train_features = []
    train_targets = []
    test_features = []
    test_targets = []
    for i in range(len(train)):
        temp_train = narrow(train[i]["lr"], 1, 1, 3)
        for i in range(len(temp_train)):
            permute = [2, 1, 0]
            train_features.append(temp_train[i][permute,:,:])
#         train_features.append(narrow(train[i]["lr"], 1, 1, 3))       
        train_targets.append(train[i]["hr"][0])
    for i in range(len(test)):
        temp_test = narrow(train[i]["lr"], 1, 1, 3)
        for i in range(len(temp_test)):
            permute = [2, 1, 0]
            test_features.append(temp_test[i][permute,:,:])
#         test_features.append(narrow(test[i]["lr"], 1, 1, 3))
        test_targets.append(test[i]["hr"][0])
    train = CustomImageDataset(train_features, train_targets)
    test = CustomImageDataset(test_features, test_targets)
    train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return train, test

# Class definition of SRCNN
# Use paddings to make sure that the output has the same shape with the input. 
class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5//2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
    
    
def train_network(network, epoch, train_loader, optimizer, log_interval,
                  train_losses, train_counter):
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        data = F.interpolate(data, size=(1024, 1024), mode='bicubic')
        output = network(data)
        save_image(output[7], 'epoch_{}_batch_{}_output.png'.format(epoch, batch_idx))
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            end = time.time()
            total_time = end - start
            start = time.time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime used: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), total_time))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 8) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), 'modelpth/model.pth')
            torch.save(optimizer.state_dict(), 'opitimizerpth/optimizer.pth')
    

def test_network(network, test_loader, test_losses):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    network.to(device)
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) 
            data = F.interpolate(data, size=(1024, 1024), mode='bicubic')
            output = network(data)
            test_loss += F.mse_loss(output, target, size_average=False).item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    end = time.time()
    total_time = end - start
    print('\nTest set: Avg. loss: {:.4f}  Time used: {:.2f}'.format(
        test_loss, total_time))
    
def main():
    torch.cuda.empty_cache()
    # Load train and test dataloaders.
    train_loader, test_loader = dataloaders()
    print('Finished loading data.')
 
    n_epochs = 5
    batch_size_train = 8
    batch_size_test =  8
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 1
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    print("Examining test data")
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)
    
    network = SRCNN()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(test_loader.dataset) for i in range(n_epochs + 1)]
    test_network(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train_network(network, epoch, train_loader, optimizer, log_interval,
                      train_losses, train_counter)
        test_network(network, test_loader, test_losses)
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    fig


if __name__ == "__main__":
    main()
    

