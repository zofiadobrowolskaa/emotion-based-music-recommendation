import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# pretrained model for transfer learning
from tensorflow.keras.applications import MobileNetV2

# keras model building components
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam


# learning curves visualization
def plot_learning_curves(history, results_dir):

    plt.figure(figsize=(12, 4))

    # accuracy subplot
    plt.subplot(1, 2, 1)

    # training accuracy over epochs
    plt.plot(history.history['accuracy'], label='train accuracy')

    # validation accuracy over epochs
    plt.plot(history.history['val_accuracy'], label='val accuracy')

    plt.title('Transfer learning - accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend()

    # loss subplot
    plt.subplot(1, 2, 2)

    # training loss over epochs
    plt.plot(history.history['loss'], label='train loss')

    # validation loss over epochs
    plt.plot(history.history['val_loss'], label='val loss')

    plt.title('Transfer learning - loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()

    # adjust layout to avoid overlap
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, 'tl_learning_curves.png'))

    plt.close()


def evaluate_tl_model(model, x_test, y_test, classes, results_dir):

    # predict class probabilities for test set
    y_pred_prob = model.predict(x_test, verbose=0)

    # convert probabilities into class indices
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\nTransfer learning evaluation")

    # generate detailed classification metrics for each class: precision, recall, f1-score
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=classes,
            zero_division=0
        )
    )

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))

    # plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,          # show numbers in cells
        fmt='d',             # integer format
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title('Confusion matrix - Transfer learning')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(os.path.join(results_dir, 'cm_transfer_learning.png'))

    plt.close()

if __name__ == "__main__":

    print("Starting transfer learning setup...")

    data_dir = "../data/processed/arrays"

    results_dir = "../results"

    os.makedirs(results_dir, exist_ok=True)

    # load dataset

    # load grayscale images (1 channel)
    x_data_gray = np.load(os.path.join(data_dir, "x_custom_aug.npy"))

    # load labels (emotion classes)
    y_data = np.load(os.path.join(data_dir, "y_custom_aug.npy"))

    # load class names from file
    with open(os.path.join(data_dir, "classes.txt"), "r") as f:
        classes = f.read().split(",")

    # convert grayscale -> pseudo RGB

    # MobileNetV2 requires 3-channel input (RGB)
    # (we duplicate grayscale channel 3 times)
    x_data_rgb = np.repeat(x_data_gray, 3, axis=-1)

    print(f"Data converted to pseudo-rgb. new shape: {x_data_rgb.shape}")


    # split dataset into training and testing subsets
    # 80% training, 20% testing
    x_train, x_test, y_train, y_test = train_test_split(
        x_data_rgb,
        y_data,
        test_size=0.2,
        random_state=42
    )

    # define model input shape
    input_shape = x_data_rgb.shape[1:]

    # number of emotion classes
    num_classes = len(classes)

    # load pretrained model

    print("Loading pre-trained mobilenetv2...")

    # load MobileNetV2 without top classification layer
    # pretrained on ImageNet dataset
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # freeze base model weights (feature extractor is not trained)
    base_model.trainable = False

    # build transfer learning model

    model = Sequential([

        # pretrained convolutional base
        base_model,

        # global pooling reduces feature maps to single vector
        GlobalAveragePooling2D(),

        # fully connected layer for learning task-specific patterns
        Dense(128, activation='relu'),

        # dropout reduces overfitting
        Dropout(0.3),

        # output layer (softmax classification)
        Dense(num_classes, activation='softmax')
    ])

    # compile model configuration
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # training phase

    print("Training new classification head...")

    history = model.fit(

        # training data
        x_train,
        y_train,

        epochs=20,

        # number of samples per gradient update
        batch_size=16,

        # validation split from training data
        validation_split=0.1,

        # show training progress
        verbose=1
    )

    plot_learning_curves(history, results_dir)

    evaluate_tl_model(model, x_test, y_test, classes, results_dir)

    print(f"\nTransfer learning finished. Plots saved to {results_dir}.")