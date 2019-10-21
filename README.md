# GAN
My first attempt with Generative Adversarial Networks. 

It allwow to create a Gan object to train a Generative adversarial network (https://en.wikipedia.org/wiki/Generative_adversarial_network).
A gan consists of a generator and a discriminator, the generator tries to trick the discriminator making it believe that the generated data is actually real. Both discriminator and generator are continuosly learning.

# How to use it

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
