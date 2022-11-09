import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import pytorch_lightning as pl
import torchmetrics




learning_rate=1e-3

def convert_resnet_to_cifar(model):
    num_features = model.fc.in_features
    model.fc=nn.Linear(num_features,10)
    return model


# models trained on imagenet
resnet18=models.resnet18(pretrained=True)
# models for ImageNet
resnet18_cifar = convert_resnet_to_cifar(resnet18)


class LitCifarResnet(pl.LightningModule):
    def __init__(self):
        super(LitCifarResnet,self).__init__()
        self.model = resnet18_cifar
        self.test_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate)
        return optimizer
    
    def training_step(self,batch,batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs,labels)
        self.log("train_loss", loss)
        return {"loss":loss}
    
    def validation_step(self,batch,batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs,labels)
        self.valid_acc(outputs, labels)
        self.log("val_loss", loss)
        self.log("val_acc",self.valid_acc)
        return {"val_loss":loss}

    def test_step(self,batch,batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs,labels)
        self.test_acc(outputs, labels)
        self.log("test_loss", loss)
        self.log('test_acc', self.test_acc)
        return {"test_loss":loss}
    
    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {"val_loss":avg_loss}
		
        return {"val_lss":avg_loss,"log":tensorboard_logs}