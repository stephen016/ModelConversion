import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from .CifarResnet_tf import Resnet18,Resnet34

model_map={
    "resnet50": ResNet50,
    "densenet121":DenseNet121,
    "inception_resnet_v2":InceptionResNetV2
}
def get_tf_model(model_name,num_classes,retrain_feature_extractor):
    base = model_map[model_name]
    base_model = base(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    if not retrain_feature_extractor:
        for layer in base_model.layers:
            layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    return model

def tf_resnet50(num_classes,retrain=False):
    model = get_tf_model("resnet50",num_classes=num_classes,retrain_feature_extractor=retrain)
    return model
def tf_densenet121(num_classes,retrain=False):
    model = get_tf_model("densenet121",num_classes=num_classes,retrain_feature_extractor=retrain)
    return model
def tf_inception_resnet_v2(num_classes,retrain):
    model = get_tf_model("inception_resnet_v2",num_classes=num_classes,retrain_feature_extractor=retrain)
    return model

def tf_resnet18(num_classes):
    resnet = Resnet18(num_classes=num_classes)
    inputs = tf.keras.layers.Input(shape=(224,224,3))
    outputs = resnet(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    return model
def tf_resnet34(num_classes):
    resnet = Resnet18(num_classes=num_classes)
    inputs = tf.keras.layers.Input(shape=(224,224,3))
    outputs = resnet(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    return model