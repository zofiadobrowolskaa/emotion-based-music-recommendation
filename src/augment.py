import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def add_noise(img):

    # this function adds small gaussian noise to an image
    # it is used as a custom augmentation step to improve model robustness

    # generate gaussian noise with mean=0 and small standard deviation
    noise = np.random.normal(loc=0.0, scale=0.03, size=img.shape)

    img_noisy = img + noise

    # clip values to keep pixels in valid range [0, 1] (prevents invalid pixel values after noise addition)
    return np.clip(img_noisy, 0.0, 1.0)


def augment_dataset(x_path, y_path, output_dir, augmentation_multiplier=3):

    # load preprocessed image dataset (features)
    print("Loading data for augmentation...")
    x_data = np.load(x_path)

    # load corresponding labels (emotion classes)
    y_data = np.load(y_path)

    # image augmentation pipeline configuration
    # this simulates real-world variations in facial images
    datagen = ImageDataGenerator(

        # small random rotations to simulate head tilt
        rotation_range=15,

        # zoom in/out to simulate different face distances
        zoom_range=0.15,

        # horizontal flipping to increase dataset variability
        horizontal_flip=True,

        # apply custom noise function defined above
        preprocessing_function=add_noise,

        # fill empty pixels created by transformations
        fill_mode='nearest'
    )

    # list for storing augmented images
    augmented_x = []

    # list for storing augmented labels
    augmented_y = []

    print("Starting augmentation process...")

    # iterate over entire dataset
    for i in range(len(x_data)):

        img = x_data[i]
        label = y_data[i]

        # keep original image in dataset (important for training stability)
        augmented_x.append(img)
        augmented_y.append(label)

        # expand dimensions because generator expects batch input
        # shape becomes: (1, height, width, channels)
        img_expanded = np.expand_dims(img, 0)

        # create augmentation generator for this image
        aug_iter = datagen.flow(img_expanded, batch_size=1)

        # generate multiple augmented versions of same image
        for _ in range(augmentation_multiplier):

            # get next augmented image from generator
            aug_img = next(aug_iter)[0]

            # store augmented image
            augmented_x.append(aug_img)

            # store same label as original image
            augmented_y.append(label)

    # convert lists to numpy arrays for efficient ML processing
    aug_x_data = np.array(augmented_x)
    aug_y_data = np.array(augmented_y)

    os.makedirs(output_dir, exist_ok=True)

    # save augmented image and labels dataset
    np.save(os.path.join(output_dir, "x_custom_aug.npy"), aug_x_data)
    np.save(os.path.join(output_dir, "y_custom_aug.npy"), aug_y_data)

    print("Augmentation completed successfully!")
    print(f"Original dataset size: {len(x_data)}")
    print(f"Augmented dataset size: {len(aug_x_data)}")


if __name__ == "__main__":

    base_dir = "../data/processed/arrays"

    x_file = os.path.join(base_dir, "x_custom.npy")
    y_file = os.path.join(base_dir, "y_custom.npy")

    if os.path.exists(x_file) and os.path.exists(y_file):

        # augmentation_multiplier controls how many augmented versions per image
        # example: 3 means each image generates 3 new ones + original
        augment_dataset(x_file, y_file, base_dir, augmentation_multiplier=3)

    else:
        print(f"Error: normalized data arrays not found in {base_dir}. Run preprocess.py first.")