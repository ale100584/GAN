# GAN
My first attempt with Generative Adversarial Networks. 

It allows to create a Gan object to train a Generative adversarial network (https://en.wikipedia.org/wiki/Generative_adversarial_network).
A gan consists of a generator and a discriminator, the generator tries to trick the discriminator into making it believe that the generated data is actually real. Both discriminator and generator are continuosly learning.

# How to use it
## Train a simple GAN
A simple experiment is training a GAN using MNIST dataset of handwritten digits. In the example below the generator takes a vector of random values as input and generates an imga of the same shape of MNIST dataset. The discriminator has to detect weather the image has been generated or it belongs to the original dataset.

### Code
It can use any keras model as generator and discriminator. They only have to comply with these constraints:
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

### Visualize results
The following code visulizes the results by loading a previously saved generator. 
```python
from GAN import utils

utils.mnist_visaulize_generator_output('generator_final.hdf5')
```
This is the result:


![Result Image](https://i.imgur.com/N8csSEu.png)

## Conditional GAN
In the example above it's impossible to tell the generator what digit we want it to generate. Conditional GANs are a type of GAN that allow condtional information to be embedded in the input of the generator.  
### Code
Loading modules and dataset:
```python
import tensorflow as tf
from GAN import gan
from GAN import generators
from GAN import discriminators
from GAN.image import char2image
import numpy as np

# MNIST dataset as usual
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_train_sample = x_train[0]
```

Creating the conditional input vector a random component and a conditional component. In this case I decided the conditional component to be a flatten image of the digit (char2image('6') returns an image of the character 6).
```python
noise_shape = (100,)
input_sample = char2image('6')
input_shape = (noise_shape[0]+input_sample.flatten().shape[0],)
```

Generator and discriminator have the same architecture as the previous example.
```python
generator = generators.mnist_conv_generator(input_shape, x_train_sample.shape)
print(generator.input.shape, generator.output.shape)
discriminator = discriminators.mnist_discriminator(x_train_sample.shape)
mnist_gan = gan.Gan(generator, discriminator,input_shape)
```

The training part is a bit different because this time every batch contains picture of one digit.
```python
epochs = 7000
noise_batch_size = 32
save_interval = 100

for epoch in range(epochs):
    # looping over digits
    for i in range(10):
        # Generating conditional part of input 
        img = char2image('{}'.format(i)).flatten()
        imgs = np.array([img] * noise_batch_size)
        
        # Generating random part of input for training the discriminator
        noise = np.random.normal(0, 1, (noise_batch_size,)+noise_shape)
        
        # Adding the two parts together to create the input for training the discriminator
        input_noise_disc = np.hstack([imgs,noise])
        
        # Generating input for training the generator 
        noise = np.random.normal(0, 1, (noise_batch_size*2,)+noise_shape)
        imgs = np.array([img] * noise_batch_size*2)
        input_noise_gen = np.hstack([imgs,noise])
        
        # Getting all the images of i-th digit and random sampling them
        x_train_filt = x_train[y_train==i]
        idx = np.random.randint(0, x_train_filt.shape[0], noise_batch_size)
        x_train_batch = x_train_filt[idx]
        
        # Train the GAN on this batch
        mnist_gan.train_on_batch(x_train_batch, input_noise_disc, input_noise_gen)

    # At save interval --> save generator and discriminator
    if epoch % save_interval == 0:
        mnist_gan.save_generator(epoch)
        mnist_gan.save_discriminator(epoch)
```

### Visualize Results
## Aknowledgments
I've done this mostly as a training excersize to learn GANs so I lost track of other repos and websites I copied from for this work, so apologies to those I copied from and didn't aknowledged.
