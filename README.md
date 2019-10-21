# GAN
My first attempt with Generative Adversarial Networks. 

It allows to create a Gan object to train a Generative adversarial network (https://en.wikipedia.org/wiki/Generative_adversarial_network).
A gan consists of a generator and a discriminator, the generator tries to trick the discriminator making it believe that the generated data is actually real. Both discriminator and generator are continuosly learning.

# How to use it
## Train a GAN
It can use any keras model as generator and discriminator. They only have to comply with these constrains:
1. The shape of the generator input has to match the noise_shape
2. The shape of the generator output has to match the discriminator input and the training set (the size of mnist image in the folllowing example)
3. The output of the discriminator has to be binary 
I included generators.py and discriminators.py to collect my models

```python
import tensorflow as tf
from GAN import gan
from GAN import generators
from GAN import discriminators

# using mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_train_sample = x_train[0]

# input shape
noise_shape=(100,)

# create generator and discriminator models
generator = generators.mnist_generator(noise_shape, x_train_sample.shape)
discriminator = discriminators.mnist_discriminator(x_train_sample.shape)

# creating the GAN
mnist_gan = gan.Gan(generator, discriminator,noise_shape)

# Traning
mnist_gan.train(x_train, epochs=10000, batch_size=128, save_interval=200)
```

## Visualize results
The following code visulizes the results by loading a previously saved generator
```python
from GAN import utils

utils.mnist_visaulize_generator_output('generator_final.hdf5')
```
This is the result:
![Result Image](https://i.imgur.com/N8csSEu.png)

## Aknowledgments
I've done this mostly as a training excersize to learn GANs so I lost track of other repos and websites I copied from for this work, so apologies to those I copied from and didn't aknowledged.
