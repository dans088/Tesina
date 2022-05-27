import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.inputs.keras import PlotLossesCallback
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from matplotlib import pyplot
from matplotlib import image


train_dir = 'train'
validation_dir = 'validation'
test_dir = 'test'


#data augmentation
train_datagen = ImageDataGenerator(
							rescale = 1./255,
							rotation_range = 40,
							width_shift_range = 0.2,
							height_shift_range = 0.2,
							shear_range = 0.2,
							zoom_range = 0.2,
                            vertical_flip=True,
							horizontal_flip = True,)

#standardize data
val_datagen = ImageDataGenerator(1./255)


#load data
train_generator = train_datagen.flow_from_directory(
							train_dir,
							target_size = (256, 256),
							batch_size = 2,
							class_mode ='binary')


val_generator = val_datagen.flow_from_directory(
							validation_dir,
							target_size = (256,256),
							batch_size =2,
							class_mode= 'binary')





def create_model(input_shape, n_classes = 1, optimizer='rmsprop', fine_tune=0):
    """
    Compiles a model integrated with VGG16 pretrained layers
    
    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """
    
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base_ = VGG16(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base_.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base_.layers:
            layer.trainable = False

    
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(n_classes,activation='sigmoid'))
    

    # Compiles the model for training.
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model


def predictImage(tile_path):
    
    img = tf.keras.utils.load_img(
        tile_path, target_size=(256, 256)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(np.argmax(score), 100 * np.max(score))
    )






input_shape = (256, 256, 3)
optim = optimizers.RMSprop(learning_rate=2e-5)


vgg_model_ft = create_model(input_shape, 1, optim, fine_tune=0)



plot_loss = PlotLossesCallback()

# Saves Keras model after each epoch - save best weights
tl_checkpoint_1 = ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5',
                                  save_best_only=True,
                                  verbose=1)

#Early stopping to prevent overtraining and to ensure decreasing validation loss
early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')


# Retrain model with fine-tuning
vgg_ft_history = vgg_model_ft.fit(train_generator,
                                  epochs=10,
                                  validation_data=val_generator,
                                  steps_per_epoch=20, 
                                  validation_steps=5,
                                  callbacks=[tl_checkpoint_1, early_stop, plot_loss],
                                  verbose=1)

# Generate predictions
vgg_model_ft.load_weights('tl_model_v1.weights.best.hdf5') # initialize the best trained weights


test_datagen = ImageDataGenerator(1./255)
test_generator = test_datagen.flow_from_directory(
					test_dir,
					target_size = (256, 256),
					batch_size = 2,
					class_mode= 'binary')

test_loss, test_acc = vgg_model_ft.evaluate(test_generator, steps = 15)
print('\ntest acc :\n', test_acc)


name = input("Enter image name: ")
img_path = 'BOTH_CLASS' + '/' + name
img = image.imread(img_path)
pyplot.imshow(img)

predictImage(img_path)
