import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, ReLU
import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential 
import tensorflow_federated as tff
# import keras.src.layers.regularization.dropout as keras_dropout_module
from keras.layers import Dropout
import os
import random
from collections import OrderedDict, defaultdict
import collections
import numpy as np



# seed = 25 
seed = 42
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
keras.utils.set_random_seed(seed)

# Πιο αυστηρό seeding
tf.config.experimental.enable_op_determinism()



def create_model_base_simple_cnn_model(input_shape = (224, 224, 1)):
    #--------------- Initial model from personal pc ----------------------
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, input_shape=input_shape, padding='same',
                               kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-5),kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

      tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1,  padding='same',  
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1,  padding='same',  
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
      tf.keras.layers.Activation('relu'),
    #   tf.keras.layers.Dropout(0.3, seed=seed),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),


      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(1e-5), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
    ])


def create_model_cnn_with_many_filters(input_shape):
    # ---------------Model from paper -------------------------- 
    # 50-40-30 - Best AUC: 0.9494, - EER: 0.0896, EER Threshold: 0.2222 - very good curve
    return tf.keras.models.Sequential([
    
      # Block 1
      tf.keras.layers.Conv2D(filters=50, kernel_size=(7,7), strides=1, input_shape=input_shape, padding='same', 
                              kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

      tf.keras.layers.Conv2D(filters=40, kernel_size=(3,3), strides=1, padding='same', 
                              kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
      tf.keras.layers.Conv2D(filters=30, kernel_size=(2,2), strides=1, padding='same', 
                              kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    #   tf.keras.layers.Conv2D(filters=30, kernel_size=(2,2), strides=1, padding='same', 
    #                           kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
    #   tf.keras.layers.Activation('relu'),
    #   tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(1e-5), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),

 ])


def create_model_cnn_with_many_blocks(input_shape):

  return tf.keras.models.Sequential([
    
      # Block 1
      tf.keras.layers.Conv2D(filters=6, kernel_size=(3,3), strides=1, input_shape=input_shape, padding='same', 
                              kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.Conv2D(filters=6, kernel_size=(3,3), strides=1,  padding='same',  
                              kernel_regularizer=tf.keras.regularizers.l2(1e-5), kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

      # Block 2
      tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1,  padding='same',  
                              kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1,  padding='same',  
                             kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.Dropout(0.3, seed=seed),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      
      # Block 3
      tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2,  padding='same',  
                              kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2,  padding='same',  
                             kernel_regularizer=tf.keras.regularizers.l2(1e-5), kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.Dropout(0.5, seed=seed),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

      # Block 4
    #   tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2,  padding='same',  
    #                           kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
    #   tf.keras.layers.Activation('relu'),
    #   tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2,  padding='same',  
    #                           kernel_regularizer=tf.keras.regularizers.l2(1e-5), kernel_initializer=tf.keras.initializers.HeUniform(seed=seed)),
    #   tf.keras.layers.Activation('relu'),
    #   tf.keras.layers.Dropout(0.5, seed=seed),
    #   tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(1e-5), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
  ])


def create_model_transfer_learning_DesNet121(input_shape):

    precision = tf.keras.mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(precision)

    # EfficientNetV2B0
    base_model = tf.keras.applications.DenseNet121(
          # weights='/code/modifier/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', 
          weights="imagenet",
          include_top=False, 
          input_shape=(224,224,3)
    )

    base_model.trainable=True
    for layer in base_model.layers[:-40]:
        layer.trainable=False

    tf.print(f"\n=== Fine-tuning top {5} layers ===")
    for layer in base_model.layers[-40:]:
        tf.print("  Trainable: {layer.name}")

    inputs = keras.Input(shape=input_shape)

    x = tf.keras.applications.densenet.preprocess_input(inputs)
    x = base_model(x, training=True)
    x = tf.keras.layers.Dropout(0.2, seed=seed)(x)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(0.15, seed=seed)(x)
    # x = tf.keras.layers.Dense(30, activation='relu',  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
    # x = tf.keras.layers.Dropout(0.2, seed=seed)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model # LR: 0.0001




def create_model_transfer_learning_ResNet50V2(input_shape):

    precision = tf.keras.mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(precision)

    # EfficientNetV2B0
    base_model = tf.keras.applications.ResNet50V2(
          # weights='/code/modifier/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', 
          # pooling='avg',
          weights="imagenet",
          include_top=False, 
          input_shape=(224,224,3)

    )

    base_model.trainable=False
    global_average_layer = tf.keras.layers.Flatten()

    hidden_fcn_layer_1 = tf.keras.layers.Dense(128, activation='relu',  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))
    hidden_fcn_layer_2 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))
    # prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))

    
    inputs = keras.Input(shape=input_shape)
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    tf.keras.layers.Lambda(lambda x: tf.debugging.check_numerics(x, "NaN/Inf detected"))
    x = global_average_layer(x)

    outputs = prediction_layer(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model # LR: 0.0001


def create_model_transfer_learning_EfficientNetV2B0(input_shape):

    precision = tf.keras.mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(precision)

    # EfficientNetV2B0
    base_model = tf.keras.applications.EfficientNetV2B0(
          # weights='/code/modifier/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', 
          # pooling='avg',
          weights="imagenet",
          include_top=False, 
          input_shape=(224,224,3)

    )

    # base_model.trainable=False
    base_model.trainable=True
    for layer in base_model.layers[:-25]:
        layer.trainable=False
    
    tf.print(f"\n=== Fine-tuning top {5} layers ===")
    for layer in base_model.layers[-25:]:
        tf.print("  Trainable: {layer.name}")
    # hidden_fcn_layer_1 = tf.keras.layers.Dense(128, activation='relu',  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))
    
    inputs = keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    x = base_model(x, training=True)
    x = tf.keras.layers.Dropout(0.2, seed=seed)(x)
    x = tf.keras.layers.Flatten()(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model # LR: 0.0001

def create_model_transfer_lerning_mobileNetV3Small(input_shape):

    precision = tf.keras.mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(precision)

    # base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
    #                                            include_top=False,
    #                                            weights='imagenet')

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        include_preprocessing=False, 
        weights='imagenet'#,
        # dropout_rate=0.2,
    )
    
    # base_model.trainable=False
    base_model.trainable=True
    for layer in base_model.layers[:-20]:
        layer.trainable=False
    
    tf.print(f"\n=== Fine-tuning top {5} layers ===")
    for layer in base_model.layers[-20:]:
        tf.print("  Trainable: {layer.name}")

    inputs = keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(x, training=True)
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2, seed=seed)(x)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(128, activation='relu',  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
    # x = tf.keras.layers.Dropout(0.2, seed=seed)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model 