import tensorflow as tf
from tensorflow.keras import layers, models

def mnist_discriminator(input_shape):
    model = models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    img = layers.Input(shape=input_shape)
    validity = model(img)

    discriminator = models.Model(img, validity)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.RMSprop(lr=0.0002, rho=0.9),
                          metrics=['accuracy'])
    return discriminator

def mnist_conv_discriminator(input_shape):
    model = models.Sequential()
    model.add(layers.Reshape(input_shape+(1,)))
    model.add(layers.Conv2D(input_shape=input_shape+(1,), filters=32, kernel_size=[5, 5], padding="same", data_format="channels_last"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=2))
    model.add(layers.Conv2D(filters=64, kernel_size=[5, 5], padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    img = layers.Input(shape=input_shape)
    validity = model(img)

    discriminator = models.Model(img, validity)
    discriminator.compile(loss='binary_crossentropy', 
                          optimizer=tf.keras.optimizers.RMSprop(lr=0.0002, rho=0.9), 
                          metrics=['accuracy'])
    return discriminator
