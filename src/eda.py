import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

dataset_path = "../data/fer2013/train"
results_path = "../results"

def plot_class_distribution(base_path):
    # get all emotion folders from dataset directory
    emotions = os.listdir(base_path)

    counts = []
    
    for emotion in emotions:
        folder_path = os.path.join(base_path, emotion)

        if os.path.isdir(folder_path):

            # count number of images inside folder
            image_count = len(os.listdir(folder_path))
            counts.append(image_count)

            print(f"{emotion}: {image_count} images")

    # create figure for visualization
    plt.figure(figsize=(10, 6))

    # create bar chart showing class balance
    sns.barplot(x=emotions, y=counts, palette="viridis")

    plt.title("Class Distribution in FER-2013 Training Set", fontsize=14)
    plt.xlabel("Emotions", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)

    plt.savefig(os.path.join(results_path, "class_distribution.png"))

    plt.show()


def show_sample_images(base_path):

    # get only valid emotion directories
    emotions = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]

    # create subplot layout
    fig, axes = plt.subplots(1, len(emotions), figsize=(15, 3))

    # iterate through emotion classes
    for idx, emotion in enumerate(emotions):

        folder_path = os.path.join(base_path, emotion)

        sample_img = os.listdir(folder_path)[0]

        img_path = os.path.join(folder_path, sample_img)

        # load image into memory
        img = mpimg.imread(img_path)

        # display image in grayscale
        axes[idx].imshow(img, cmap="gray")

        # add emotion label above image
        axes[idx].set_title(emotion)

        # hide axis values for cleaner visualization
        axes[idx].axis("off")

    plt.suptitle("Sample Images for Each Emotion", fontsize=16)

    plt.savefig(os.path.join(results_path, "sample_images.png"))

    plt.show()


if __name__ == "__main__":
    print("Starting eda process...")

    # create results directory if it does not exist
    os.makedirs(results_path, exist_ok=True)

    if not os.path.exists(dataset_path):
        print(f"Error: dataset path '{dataset_path}' not found.")

    else:
        plot_class_distribution(dataset_path)

        show_sample_images(dataset_path)

        print("EDA completed successfully. Plots saved to 'results' folder.")