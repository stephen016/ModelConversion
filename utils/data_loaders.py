import torch
import torchvision
import torchvision.transforms as transforms
from timm.data import ImageDataset
from timm.data.transforms_factory import create_transform


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

def get_imagenette_loader(batch_size):
    train_dataset=ImageDataset("./data/imagenette2/train",transform=create_transform(224))
    test_set = ImageDataset("./data/imagenette2/val",transform=create_transform(224))
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