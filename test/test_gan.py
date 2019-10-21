import pytest
from GAN import gan, generators, discriminators

def test_gan():
    g_output_shape = (28,28)
    input_shape = (100,)
    g = generators.mnist_generator(input_shape, g_output_shape)
    d = discriminators.mnist_discriminator(g_output_shape)
    good_gan = gan.Gan(g,d,input_shape)

    # Cheking that mismatching input_shape between gan and generator cause exception throw
    with pytest.raises(Exception):
        g = generators.mnist_generator((99,), g_output_shape)
        gan.Gan(g,d,input_shape)

    # Checking that mismatching generator output - discriminator input cause exception throw
    with pytest.raises(Exception):
        g = generators.mnist_generator(input_shape, g_output_shape)
        d = discriminators.mnist_discriminator((29,29))
        gan.Gan(g,d,input_shape)
