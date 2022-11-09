import torch
import torchvision
import torchvision.transforms as transforms


# CIFAR10

def get_cifar_loader(batch_size):
    train_dataset = torchvision.datasets.CIFAR10(root="./data",
                                                train=True,
                                                transform=transforms.ToTensor(),
                                                download=True)
    test_set = torchvision.datasets.CIFAR10(root="./data",
                                                train=False,
                                                transform=transforms.ToTensor(),
                                                download=True)
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size = batch_size,
                                               num_workers=4,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                             batch_size = batch_size,
                                             num_workers=4,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size = batch_size,
                                              num_workers=4,
                                              shuffle=False)
    return train_loader,val_loader,test_loader