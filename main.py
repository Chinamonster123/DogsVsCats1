import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from dataset import CatDogDataset
from utils import split, train

if __name__ == '__main__':
    # input train data and test data
    train_dir = './input/train'
    val_dir = './input/val'
    test_dir = './input/test1'

    split.split_train_test(train_dir, val_dir)

    train_files = os.listdir(train_dir)
    val_files = os.listdir(val_dir)
    test_files = os.listdir(test_dir)

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor()
    ])

    cat_files = [tf for tf in train_files if 'cat' in tf]
    dog_files = [tf for tf in train_files if 'dog' in tf]

    cats = CatDogDataset(cat_files, train_dir, transform=data_transform)
    dogs = CatDogDataset(dog_files, train_dir, transform=data_transform)

    catdogs = ConcatDataset([cats, dogs])

    dataloader = DataLoader(catdogs, batch_size=32, shuffle=True, num_workers=4)

    samples, labels = iter(dataloader).next()
    plt.figure(figsize=(16, 24))
    grid_imgs = torchvision.utils.make_grid(samples[:24])
    np_grid_imgs = grid_imgs.numpy()
    # in tensor, image is (batch, width, height), so you have to transpose it to (width, height, batch) in numpy to show it.
    plt.imshow(np.transpose(np_grid_imgs, (1, 2, 0)))

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # val_dataset
    valset = CatDogDataset(val_files, val_dir, mode='test', transform=test_transform)
    valloader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=4)

    # test_dataset
    testset = CatDogDataset(test_files, test_dir, mode='test', transform=test_transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    # transfer learning

    # use GPU
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    print(device)

    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        nn.Linear(500, 2)
    )

    model = model.to(device)

    ## train parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.5)
    num_epochs = 1

    def main():
        trainer = train.Trainer(criterion, optimizer, model)
        trainer.loop(num_epochs, dataloader, valloader)
        trainer.test(device, testloader)

    main()
