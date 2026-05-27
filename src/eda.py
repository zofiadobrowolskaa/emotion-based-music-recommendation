import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
import numpy as np

dataset_path = "../data/fer2013/train"
custom_dataset_path = "../data/processed/custom_dataset"
results_path = "../results"


def plot_class_distribution(base_path, title_suffix="", filename="class_distribution.png"):

    # get all emotion folders from dataset directory
    emotions = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]

    counts = []

    for emotion in emotions:
        folder_path = os.path.join(base_path, emotion)
        if os.path.isdir(folder_path):
            image_count = len([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            counts.append(image_count)
            print(f"{emotion}: {image_count} images")

    total = sum(counts)

    # compute percentage share of each class in the dataset
    percentages = [c / total * 100 for c in counts]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # bar chart of absolute counts
    # hue=emotions + legend=False is the new seaborn API for per-bar coloring
    sns.barplot(x=emotions, y=counts, hue=emotions, palette="viridis", legend=False, ax=axes[0])
    axes[0].set_title(f"Class Distribution {title_suffix} (absolute count)", fontsize=13)
    axes[0].set_xlabel("Emotion", fontsize=11)
    axes[0].set_ylabel("Number of Images", fontsize=11)

    # add count labels on top of each bar
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        axes[0].text(i, count + total * 0.005, f"{count}\n({pct:.1f}%)",
                     ha='center', va='bottom', fontsize=9)

    # pie chart showing percentage distribution (helps assess class imbalance visually)
    axes[1].pie(
        counts,
        labels=emotions,
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette("viridis", len(emotions))
    )
    axes[1].set_title(f"Class Balance {title_suffix} (percentage)", fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, filename), dpi=150)
    plt.close()

    # save numeric stats to csv for use in the report
    stats_df = pd.DataFrame({
        'Emotion': emotions,
        'Count': counts,
        'Percentage (%)': [round(p, 2) for p in percentages]
    })

    stats_filename = filename.replace('.png', '_stats.csv')
    stats_df.to_csv(os.path.join(results_path, stats_filename), index=False)

    print(f"\nClass distribution stats:")
    print(stats_df.to_string(index=False))
    print(f"Total images: {total}")
    print(f"Most common class: {emotions[np.argmax(counts)]} ({max(counts)} images)")
    print(f"Least common class: {emotions[np.argmin(counts)]} ({min(counts)} images)")
    print(f"Imbalance ratio (max/min): {max(counts)/min(counts):.2f}x")

    return stats_df


def show_sample_images(base_path, filename="sample_images.png"):

    # get only valid emotion directories
    emotions = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]

    # create subplot layout: one image per emotion class
    fig, axes = plt.subplots(1, len(emotions), figsize=(3 * len(emotions), 4))

    if len(emotions) == 1:
        axes = [axes]

    for idx, emotion in enumerate(emotions):

        folder_path = os.path.join(base_path, emotion)

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not image_files:
            continue

        sample_img = image_files[0]
        img_path = os.path.join(folder_path, sample_img)

        # load image into memory
        img = mpimg.imread(img_path)

        # display image in grayscale
        axes[idx].imshow(img, cmap="gray")

        # add emotion label as subplot title
        axes[idx].set_title(emotion, fontsize=10)

        # hide axis ticks for cleaner visualization
        axes[idx].axis("off")

    plt.suptitle("Sample Images for Each Emotion Class", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, filename), dpi=150)
    plt.close()


def compare_datasets_bar(fer_stats, custom_stats):
    # side-by-side bar chart comparing fer-2013 and custom dataset distributions
    # helps visualize how well the custom dataset mirrors the original distribution

    # find shared emotion classes
    emotions = fer_stats['Emotion'].tolist()

    x = np.arange(len(emotions))
    width = 0.35

    # normalize to percentages for fair comparison (datasets have different total sizes)
    fer_pct = fer_stats['Percentage (%)'].tolist()
    custom_pct = custom_stats.set_index('Emotion').reindex(emotions)['Percentage (%)'].fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, fer_pct, width, label='FER-2013', color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + width/2, custom_pct, width, label='Custom Dataset', color='darkorange', alpha=0.85)

    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Percentage of Dataset (%)', fontsize=12)
    ax.set_title('Class Distribution: FER-2013 vs Custom Dataset', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "dataset_comparison.png"), dpi=150)
    plt.close()

    print("\nDataset comparison chart saved.")


if __name__ == "__main__":
    print("Starting eda process...")

    os.makedirs(results_path, exist_ok=True)

    if not os.path.exists(dataset_path):
        print(f"Error: dataset path '{dataset_path}' not found.")
    else:
        print("\n--- FER-2013 Dataset ---")
        fer_stats = plot_class_distribution(
            dataset_path,
            title_suffix="(FER-2013)",
            filename="class_distribution.png"
        )

        show_sample_images(dataset_path, filename="sample_images.png")

    # analyze custom dataset if it exists
    if os.path.exists(custom_dataset_path):
        print("\n--- Custom Dataset ---")
        custom_stats = plot_class_distribution(
            custom_dataset_path,
            title_suffix="(Custom Dataset)",
            filename="custom_class_distribution.png"
        )

        show_sample_images(custom_dataset_path, filename="custom_sample_images.png")

        # generate side-by-side comparison only when both datasets are available
        if os.path.exists(dataset_path):
            compare_datasets_bar(fer_stats, custom_stats)
    else:
        print(f"Custom dataset path not found at '{custom_dataset_path}'. Skipping custom EDA.")

    print("\nEDA completed successfully. Plots saved to 'results' folder.")
