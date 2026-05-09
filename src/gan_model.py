import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# keras functional API components
from tensorflow.keras import layers, Model


# generator network
def build_generator(latent_dim, num_classes):

    # generator input 1:
    # random noise vector sampled from latent space
    noise = layers.Input(shape=(latent_dim,))

    # generator input 2:
    # emotion label used for conditional generation
    label = layers.Input(shape=(1,), dtype='int32')

    # label embedding

    # convert integer label into dense vector representation
    label_embedding = layers.Embedding(num_classes, latent_dim)(label)

    # flatten embedding to 1d vector
    label_embedding = layers.Flatten()(label_embedding)

    # combine noise vector with embedded label
    # this conditions generation on selected emotion
    model_input = layers.multiply([noise, label_embedding])

    # dense projection

    # project latent vector into larger feature map
    # foundation for future image generation
    x = layers.Dense(12 * 12 * 128, use_bias=False)(model_input)

    # normalize activations for training stability
    x = layers.BatchNormalization()(x)

    # leakyrelu helps gradient flow
    x = layers.LeakyReLU()(x)

    # reshape vector into feature map
    # shape becomes: (12, 12, 128)
    x = layers.Reshape((12, 12, 128))(x)

    # first upsampling block

    # transpose convolution increases spatial resolution
    # output shape: 24x24
    x = layers.Conv2DTranspose(
        64,
        (5, 5),
        strides=(2, 2),
        padding='same',
        use_bias=False
    )(x)

    # normalize activations
    x = layers.BatchNormalization()(x)

    # activation function
    x = layers.LeakyReLU()(x)

    # =========================
    # second upsampling block
    # =========================

    # final transpose convolution
    # output shape: 48x48x1
    x = layers.Conv2DTranspose(
        1,
        (5, 5),
        strides=(2, 2),
        padding='same',
        use_bias=False,
        activation='sigmoid'
    )(x)

    # create keras model object
    return Model([noise, label], x, name="generator")


# discriminator network

def build_discriminator(img_shape, num_classes):

    # discriminator input 1:
    # real or generated image
    img_input = layers.Input(shape=img_shape)

    # discriminator input 2:
    # emotion label for conditional discrimination
    label = layers.Input(shape=(1,), dtype='int32')

    # label embedding

    # embed label into vector matching image size
    label_embedding = layers.Embedding(
        num_classes,
        np.prod(img_shape)
    )(label)

    # reshape embedding into image dimensions
    label_embedding = layers.Reshape(img_shape)(label_embedding)

    # concatenate image and label channels together
    # discriminator now sees both image and target emotion
    merged_input = layers.Concatenate(axis=-1)(
        [img_input, label_embedding]
    )

    # first downsampling block

    # convolution extracts features and downsamples image
    x = layers.Conv2D(
        64,
        (5, 5),
        strides=(2, 2),
        padding='same'
    )(merged_input)

    # activation function
    x = layers.LeakyReLU()(x)

    # dropout regularization
    x = layers.Dropout(0.3)(x)

    # second downsampling block

    # deeper feature extraction
    x = layers.Conv2D(
        128,
        (5, 5),
        strides=(2, 2),
        padding='same'
    )(x)

    # activation function
    x = layers.LeakyReLU()(x)

    # regularization
    x = layers.Dropout(0.3)(x)

    # classification head

    # flatten feature maps into vector
    x = layers.Flatten()(x)

    # output probability:
    # 1 = real image
    # 0 = fake image
    out = layers.Dense(1, activation='sigmoid')(x)

    # create keras model object
    return Model([img_input, label], out, name="discriminator")


# save generated image samples

def save_generated_images(generator, epoch, latent_dim, classes, results_dir):

    # number of emotion classes
    num_classes = len(classes)

    # generate random latent vectors
    noise = tf.random.normal([num_classes, latent_dim])

    # create one label for each emotion class
    labels = tf.constant([[i] for i in range(num_classes)])

    # generate fake images using generator
    generated_images = generator([noise, labels], training=False)

    plt.figure(figsize=(10, 2))

    # display one generated image per class
    for i in range(num_classes):

        plt.subplot(1, num_classes, i + 1)

        # display grayscale generated image
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')

        plt.title(classes[i], fontsize=10)

        plt.axis('off')

    # improve spacing between plots
    plt.tight_layout()

    # save generated samples for current epoch
    plt.savefig(
        os.path.join(
            results_dir,
            f'gan_generated_epoch_{epoch:03d}.png'
        )
    )

    plt.close()


if __name__ == "__main__":

    print("Starting conditional DCGAN training process...")

    data_dir = "../data/processed/arrays"

    results_dir = "../results/gan_progress"

    os.makedirs(results_dir, exist_ok=True)

    # load dataset

    # load image dataset
    x_data = np.load(os.path.join(data_dir, "x_custom_aug.npy"))

    # load emotion labels
    y_data = np.load(os.path.join(data_dir, "y_custom_aug.npy"))

    # load class names
    with open(os.path.join(data_dir, "classes.txt"), "r") as f:
        classes = f.read().split(",")

    # configuration

    # number of emotion classes
    num_classes = len(classes)

    # image shape: (48, 48, 1)
    img_shape = x_data.shape[1:]

    # size of random latent vector
    latent_dim = 100

    batch_size = 16

    epochs = 50

    # dataset preparation

    # convert numpy arrays into tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

    # shuffle dataset for randomness
    dataset = dataset.shuffle(buffer_size=1024)

    # split dataset into batches
    dataset = dataset.batch(batch_size)

    # initialize models

    # create generator model
    generator = build_generator(latent_dim, num_classes)

    # create discriminator model
    discriminator = build_discriminator(img_shape, num_classes)

    # training utilities

    # binary crossentropy loss:
    # used for real vs fake classification
    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    # optimizer for generator
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # optimizer for discriminator
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # custom training step

    # tf.function compiles function into tensorflow graph for speed
    @tf.function
    def train_step(images, labels):

        # get actual batch size dynamically
        current_batch_size = tf.shape(images)[0]

        # generate random latent vectors
        noise = tf.random.normal([current_batch_size, latent_dim])

        # record operations for automatic differentiation
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # generate fake images

            generated_images = generator(
                [noise, labels],
                training=True
            )

            # discriminator predictions

            # prediction on real images
            real_output = discriminator(
                [images, labels],
                training=True
            )

            # prediction on fake generated images
            fake_output = discriminator(
                [generated_images, labels],
                training=True
            )

            # generator loss

            # generator tries to fool discriminator
            # wants fake images classified as real
            gen_loss = cross_entropy(
                tf.ones_like(fake_output),
                fake_output
            )

            # discriminator loss

            # discriminator should classify real images as 1
            disc_loss_real = cross_entropy(
                tf.ones_like(real_output),
                real_output
            )

            # discriminator should classify fake images as 0
            disc_loss_fake = cross_entropy(
                tf.zeros_like(fake_output),
                fake_output
            )

            # total discriminator loss
            disc_loss = disc_loss_real + disc_loss_fake

        # gradient computation

        # compute generator gradients
        gradients_of_generator = gen_tape.gradient(
            gen_loss,
            generator.trainable_variables
        )

        # compute discriminator gradients
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss,
            discriminator.trainable_variables
        )

        # weight updates

        # apply generator gradients
        generator_optimizer.apply_gradients(
            zip(
                gradients_of_generator,
                generator.trainable_variables
            )
        )

        # apply discriminator gradients
        discriminator_optimizer.apply_gradients(
            zip(
                gradients_of_discriminator,
                discriminator.trainable_variables
            )
        )

        # return losses for monitoring
        return gen_loss, disc_loss


    # training loop

    print(f"Training GAN for {epochs} epochs. This might take a few minutes on cpu...")

    for epoch in range(1, epochs + 1):

        # accumulators for average losses
        gen_loss_avg = 0
        disc_loss_avg = 0
        batches = 0

        # iterate through dataset batches
        for image_batch, label_batch in dataset:

            # run one training step
            g_loss, d_loss = train_step(image_batch, label_batch)

            # accumulate losses
            gen_loss_avg += g_loss
            disc_loss_avg += d_loss

            # count processed batches
            batches += 1

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Gen Loss: {gen_loss_avg/batches:.4f} | "
            f"Disc Loss: {disc_loss_avg/batches:.4f}"
        )

        # save generated samples

        # save images every 10 epochs
        # also save final epoch output
        if epoch % 10 == 0 or epoch == epochs:
            save_generated_images(
                generator,
                epoch,
                latent_dim,
                classes,
                results_dir
            )

    # save trained weights

    generator.save_weights(
        os.path.join(
            results_dir,
            "generator_weights.weights.h5"
        )
    )

    print(
        f"GAN training complete. "
        f"Check '{results_dir}' for generated faces and model weights."
    )