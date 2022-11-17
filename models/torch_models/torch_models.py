import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import timm


learning_rate=1e-3


class LitModel(pl.LightningModule):
    def __init__(self,model_name,num_classes):
        super(LitModel,self).__init__()
        self.model=timm.create_model(model_name,pretrained=True,num_classes=num_classes)
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


def resnet18(num_classes):
    return LitModel("resnet18",num_classes)
def resnet50(num_classes):
    return LitModel("resnet50",num_classes)
def densenet121(num_classes):
    return LitModel("densenet121",num_classes)
def inception_v4(num_classes):
    return LitModel("inception_v4",num_classes)
def inception_resnet_v2(num_classes):
    return LitModel("inception_resnet_v2",num_classes)
def visformer_small(num_classes):
    return LitModel("visformer_small",num_classes)