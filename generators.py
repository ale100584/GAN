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


