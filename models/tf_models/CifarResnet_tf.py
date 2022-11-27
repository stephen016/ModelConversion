import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

# retrun a complied model
# Direclty use pretrained resnet50
   
def feature_extractor(inputs):
    feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
    return feature_extractor
def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(1024, activation="relu")(x)
    #x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x
def final_model(inputs):
    resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)
    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)
    return classification_output
def get_CifarResnet50():
    inputs = tf.keras.layers.Input(shape=(32,32,3))
    output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.001,
                                                  beta_1 = 0.9,
                                                  beta_2 = 0.999, 
                                                  amsgrad = False), 
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy'])
  
    return model

"""
 Resnet 18 and resnet 34 doesn't have official pretrained model. 
 need to define and training from scratch

"""

class BasicBlock(tf.keras.Model):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out

class BottleNeck(tf.keras.Model):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, strides=1):
        super(BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion*out_channels, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out


class BuildResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes):
        super(BuildResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], strides=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], strides=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], strides=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], strides=2)
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def _make_layer(self, block, out_channels, num_blocks, strides):
        stride = [strides] + [1]*(num_blocks-1)
        layer = []
        for s in stride:
            layer += [block(self.in_channels, out_channels, s)]
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layer)

def Resnet18(num_classes):
    return BuildResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def Resnet34(num_classes):
    return BuildResNet(BasicBlock, [3, 4, 6, 3], num_classes)


"""
# resnet 50 has pretrained version
def Resnet50(num_classes):
    return BuildResNet(BottleNeck, [3, 4, 6, 3], num_classes)

def Resnet101(num_classes):
    return BuildResNet(BottleNeck, [3, 8, 36, 3], num_classes)
 """

def get_CifarResnet18(num_classes):
    resnet = Resnet18(num_classes=num_classes)
    inputs = tf.keras.layers.Input(shape=(32,32,3))
    outputs = resnet(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.001,
                                                  beta_1 = 0.9,
                                                  beta_2 = 0.999, 
                                                  amsgrad = False), 
                loss='categorical_crossentropy',
                metrics = ['accuracy'])
  
    return model

def get_CifarResnet34(num_classes):
    resnet = Resnet34(num_classes=num_classes)
    inputs = tf.keras.layers.Input(shape=(32,32,3))
    outputs = resnet(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.001,
                                                  beta_1 = 0.9,
                                                  beta_2 = 0.999, 
                                                  amsgrad = False), 
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy'])
  
    return model


