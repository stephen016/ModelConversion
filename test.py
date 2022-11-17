import torch
from models.torch_models.torch_models import resnet18
from utils.data_loaders import get_cifar_loader,get_imagenette_loader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

if __name__=="__main__":
    print(torch.__version__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("use device:",device)
    #train_loader,val_loader,test_loader = get_cifar_loader(batch_size=256)
    train_loader,val_loader,test_loader = get_imagenette_loader(batch_size=8)
    epochs=3
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=5, verbose=False, mode="max")

    trainer = Trainer(max_epochs=epochs,fast_dev_run=False,accelerator="gpu",callbacks=[early_stop_callback])
    model = resnet18(num_classes=10).to(device)

    trainer.fit(model,train_loader,val_loader)