import os
import cv2
import numpy as np


def load_and_preprocess_data(dataset_path, target_size=(48, 48)):

    # list to store processed image data
    images = []
    # list to store corresponding labels (emotion classes as integers)
    labels = []

    if not os.path.exists(dataset_path):
        print(f"Error: Path {dataset_path} does not exist.")

        # return empty arrays if dataset is missing
        return np.array([]), np.array([]), []

    # get all emotion class folders from dataset directory
    emotions = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    print(f"Found classes: {emotions}")

    # iterate over each emotion folder with index as label
    for label_idx, emotion in enumerate(emotions):

        folder_path = os.path.join(dataset_path, emotion)

        # iterate through all images in current emotion folder
        for filename in os.listdir(folder_path):

            img_path = os.path.join(folder_path, filename)

            # read image in grayscale mode
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:

                # resize image to fixed size required by model
                img_resized = cv2.resize(img, target_size)

                # normalize pixel values from [0, 255] to [0, 1]
                # improves neural network training stability
                img_normalized = img_resized.astype('float32') / 255.0

                images.append(img_normalized)
                labels.append(label_idx)

    # convert Python lists to NumPy arrays for ML processing
    x_data = np.array(images)
    y_data = np.array(labels)

    # add channel dimension required by CNN models
    # shape becomes: (samples, 48, 48, 1)
    x_data = np.expand_dims(x_data, axis=-1)

    # return image data, labels and class names
    return x_data, y_data, emotions


if __name__ == "__main__":

    print("Starting cleaning and normalization process...")

    input_data_path = "../data/processed/custom_dataset"
    output_dir = "../data/processed/arrays"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing dataset from: {input_data_path}")

    x_custom, y_custom, classes = load_and_preprocess_data(input_data_path)

    # check if any data was loaded successfully
    if len(x_custom) > 0:

        # save feature array (images)
        np.save(os.path.join(output_dir, "x_custom.npy"), x_custom)

        # save label array (emotions)
        np.save(os.path.join(output_dir, "y_custom.npy"), y_custom)

        # save class mapping to text file for later reference
        with open(os.path.join(output_dir, "classes.txt"), "w") as f:
            f.write(",".join(classes))

        print(f"Successfully normalized and saved {len(x_custom)} images.")
        print(f"Shape of x: {x_custom.shape}")
        print(f"Range of pixel values: min={np.min(x_custom)}, max={np.max(x_custom)}")

    else:
        print("No data processed. Check your input folder.")