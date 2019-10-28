from tensorflow.keras import layers, models
import numpy as np

def mnist_generator(noise_shape, output_shape):
    model = models.Sequential()
    model.add(layers.Dense(256, input_shape=noise_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    
    
    model.add(layers.Dense(np.prod(output_shape), activation='sigmoid'))
    model.add(layers.Reshape(output_shape))

    noise = layers.Input(shape=noise_shape)
    img = model(noise)

    generator = models.Model(noise, img)
    generator.compile(loss='binary_crossentropy', optimizer='adam')
    return generator

def mnist_conv_generator(noise_shape, output_shape):
    model = models.Sequential()
    model.add(layers.Dense(28*28*16, input_shape=noise_shape)) 
    # It shouldn't be necessary to specify input_shape here but if I don't it sometimes generates an error. 
    # It seems to be an issue with some versions of tensorflow https://github.com/tensorflow/tensorflow/issues/30892.
    model.add(layers.Reshape((28,28,16)))
    
    model.add(layers.Conv2DTranspose(filters=64, kernel_size=[3,3], strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(filters=32, kernel_size=[3,3], strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(filters=16, kernel_size=[3,3], strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(filters=1, kernel_size=[3,3], strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # This layer forces the output shape to be output_shape
    model.add(layers.Flatten())
    model.add(layers.Dense(np.prod(output_shape), activation='sigmoid'))
    model.add(layers.Reshape(output_shape))
    
    noise = layers.Input(shape=noise_shape)
    img = model(noise)

    generator = models.Model(noise, img)
    generator.compile(loss='binary_crossentropy', optimizer='adam')
    return generator
