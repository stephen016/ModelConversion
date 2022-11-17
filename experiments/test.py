import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import seml
from sacred import Experiment
import torch
from models.torch_models.torch_models import resnet18
from utils.data_loaders import get_cifar_loader,get_imagenette_loader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer


ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(arguments):
    print(torch.__version__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("use device:",device)
    batch_size=arguments["batch_size"]
    #train_loader,val_loader,test_loader = get_cifar_loader(batch_size=256)
    train_loader,val_loader,test_loader = get_imagenette_loader(batch_size=batch_size)
    epochs=arguments["epoch"]
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=5, verbose=False, mode="max")

    trainer = Trainer(max_epochs=epochs,fast_dev_run=False,accelerator="gpu",callbacks=[early_stop_callback])
    model = resnet18(num_classes=10).to(device)

    trainer.fit(model,train_loader,val_loader)