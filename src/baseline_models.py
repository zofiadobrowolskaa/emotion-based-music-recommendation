import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# train-test split utility from sklearn
from sklearn.model_selection import train_test_split

# simple baseline models used for comparison
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# evaluation metrics for classification tasks
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(y_true, y_pred, classes, model_name, results_dir):

    print(f"\n{model_name} evaluation")

    # generate detailed classification metrics for each class: precision, recall, f1-score
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=classes,
            zero_division=0
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    # plot heatmap version of confusion matrix
    # diagonal = correct predictions
    # off-diagonal = misclassifications
    sns.heatmap(
        cm,
        annot=True,          # show numeric values in cells
        fmt='d',             # integer formatting
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title(f'Confusion matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plot_path = os.path.join(
        results_dir,
        f'cm_{model_name.replace(" ", "_").lower()}.png'
    )

    plt.savefig(plot_path)

    plt.close()


if __name__ == "__main__":

    print("Starting baseline models training...")

    # directory containing processed numpy arrays
    data_dir = "../data/processed/arrays"

    results_dir = "../results"

    os.makedirs(results_dir, exist_ok=True)

    # load augmented image dataset (features)
    x_data = np.load(os.path.join(data_dir, "x_custom_aug.npy"))

    # load corresponding labels (emotion classes)
    y_data = np.load(os.path.join(data_dir, "y_custom_aug.npy"))

    # load class names (emotion labels as strings)
    with open(os.path.join(data_dir, "classes.txt"), "r") as f:
        classes = f.read().split(",")

    # flatten images from 3D/4D format into 1D vectors
    # required for classical ML models (kNN, decision tree)
    n_samples = x_data.shape[0]

    # reshape: (samples, 48, 48, 1) -> (samples, 2304)
    x_flattened = x_data.reshape((n_samples, -1))

    print(f"Data flattened. new shape: {x_flattened.shape}")

    # split dataset into training and testing subsets
    # 80% training, 20% testing
    x_train, x_test, y_train, y_test = train_test_split(
        x_flattened,
        y_data,
        test_size=0.2,
        random_state=42  # ensures reproducibility of results
    )

    print("Training k-nearest neighbors...")

    # initialize kNN classifier with k=3 neighbors
    knn = KNeighborsClassifier(n_neighbors=3)

    # train model on training data
    knn.fit(x_train, y_train)

    # predict labels for test set
    y_pred_knn = knn.predict(x_test)

    evaluate_model(y_test, y_pred_knn, classes, "kNN", results_dir)

    print("Training decision tree...")

    # initialize decision tree classifier
    dt = DecisionTreeClassifier(random_state=42)

    # train model
    dt.fit(x_train, y_train)

    # make predictions
    y_pred_dt = dt.predict(x_test)

    evaluate_model(y_test, y_pred_dt, classes, "Decision Tree", results_dir)

    print(f"\nEvaluation completed. Confusion matrices saved to {results_dir}.")