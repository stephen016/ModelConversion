import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir))
sys.path.append(PROJECT_ROOT)

import seml
from sacred import Experiment
import torch
from models.torch_models.torch_models import LitModel
from utils.data_loaders import get_cifar_loader,get_imagenette_loader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import onnxruntime as ort
from onnx2pytorch import ConvertModel


ex = Experiment()
seml.setup_logger(ex)

def get_torch_acc(model,test_loader,device):
    model = model.to(device)
    model.eval()
    _correct=0
    _all=0
    for imgs,labels in test_loader:
        pred = torch.argmax(model(imgs.to(device)),axis=1).to("cpu")
        _all+=len(labels)
        _correct+=(pred==labels).sum()
    print(f"torch model accuracy:{_correct/_all}")
    return _correct/_all

def get_onnx_acc(ort_sess,test_loader):
    _correct=0
    _all=0
    for imgs,labels in test_loader:
        output = ort_sess.run(output_names=['output'],input_feed={'input': imgs.numpy()})
        pred = np.argmax(output[0],axis=1)
        _all+=len(labels)
        _correct+=(pred==labels.numpy()).sum()
    print(f"onnx model accuracy: {_correct/_all}")
    return _correct/_all

def get_tf_acc(loaded,test_loader):
    infer = loaded.signatures["serving_default"]
    key=list(infer.structured_outputs.keys())[0]
    _all=0
    _correct=0
    for imgs,labels in test_loader:
        out = infer(**{'input': imgs})
        pred = np.argmax(out[key],axis=1)
        _all+=len(labels)
        _correct+=(pred==labels.numpy()).sum()
    print(f"tensorflow model accuracy: {_correct/_all}")
    return _correct/_all

def get_acc_from_converted_pytorch_model(model,test_loader):
    model.eval()
    _all=0
    _correct=0
    for imgs,labels in test_loader:
        _all+=len(labels)
        for img,label in zip(imgs,labels):
            output = model(img.unsqueeze(0))
            _correct+=(torch.argmax(output)==label).item()
    print(f"converted torch model accuracy: {_correct/_all}")
    return _correct/_all

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
    print(f"using argument {arguments}")
    print(torch.__version__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("use device:",device)
    # step1: get model and dataset
    model_name = arguments["model"]
    num_classes = arguments["num_classes"]
    batch_size = arguments["batch_size"]
    model = LitModel(model_name=model_name,num_classes=num_classes).to(device)  
    
    if arguments["dataset"]=="cifar10":
        train_loader,val_loader,test_loader = get_cifar_loader(batch_size=batch_size)
    elif arguments["dataset"]=="imagenette":
        train_loader,val_loader,test_loader = get_imagenette_loader(batch_size=batch_size)
    
    # step2: train model if needed
    if arguments["train"]:
        # train model
        epochs=arguments["epochs"]
        early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=5, verbose=False, mode="max")
        trainer = Trainer(max_epochs=epochs,fast_dev_run=False,accelerator="gpu",callbacks=[early_stop_callback])
        trainer.fit(model,train_loader,val_loader)
        torch_model = model.model
    else:
        # load model
        load_path = arguments["load_path"]
        torch_model = model.model
        torch_model.load_state_dict(torch.load(load_path))
    # step3: save and test model accuracy 
    original_acc=get_torch_acc(torch_model,test_loader,device)
    
    torch_save_path = f"saved_models/torch2tf/{model_name}.pth"
    torch.save(torch_model.state_dict(),torch_save_path)
    
    # step4: convert to onnx and save
    if arguments["dataset"]=="cifar10":
        dummy_input = torch.randn(1,3,32,32,device="cuda")
    elif arguments["dataset"]=="imagenette":
        dummy_input = torch.randn(1,3,224,224,device="cuda")
    onnx_save_path = f"saved_models/torch2tf/{model_name}.onnx"
    torch.onnx.export(torch_model,
                  dummy_input,
                  onnx_save_path,
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={'input':{0:'batch_size'}})
    # step5: run onnx 
    ort_sess = ort.InferenceSession(onnx_save_path)
    
    onnx_acc = get_onnx_acc(ort_sess,test_loader)
    
    # step6: convert from onnx to tensorflow 
    onnx_model = onnx.load(onnx_save_path)
    tf_rep = prepare(onnx_model)
    # step7: save tf model, load and test accuracy
    tf_save_path = f"saved_models/torch2tf/{model_name}"
    tf_rep.export_graph(tf_save_path)
    
    tf_model = tf.saved_model.load(tf_save_path)
    tf_acc = get_tf_acc(tf_model,test_loader)
    
    
    # step8 covnert from onnx back to torch
    convert_torch_model = ConvertModel(onnx_model,debug=False)
    # step9 test torch model accuracy
    converted_torch_acc = get_acc_from_converted_pytorch_model(convert_torch_model,test_loader)
    
    
    print(f"original torch model acc: {original_acc}")
    print(f"onnx model acc: {onnx_acc}")
    print(f"tensorflow model acc: {tf_acc}")
    print(f"covnerted torch model acc: {converted_torch_acc}")
