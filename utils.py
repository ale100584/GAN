from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

def mnist_visaulize_generator_output(modelname, seed=100, noise_shape=(100,), rows=5, cols=5):
    loaded_generator = tf.keras.models.load_model(modelname)
    np.random.seed(seed)
    noise = np.random.normal(0, 1, (rows*cols, noise_shape[0]))
    imgs = loaded_generator.predict(noise)
    fig, axs = plt.subplots(rows, cols)

    cnt = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(imgs[cnt, :, :], cmap=plt.cm.Greys)
            axs[i, j].axis('off')
            cnt += 1
