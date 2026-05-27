import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# train-test split utility from sklearn
from sklearn.model_selection import train_test_split

# shared utility for saving metrics across all scripts
from save_metrics import save_metrics

# evaluation metrics for classification tasks
from sklearn.metrics import classification_report, confusion_matrix

# tensorflow / keras components for building cnn model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


# learning curves visualization
def plot_learning_curves(history, results_dir):

    plt.figure(figsize=(12, 4))

    # accuracy plot
    plt.subplot(1, 2, 1)

    # training accuracy over epochs
    plt.plot(history.history['accuracy'], label='train accuracy')

    # validation accuracy over epochs
    plt.plot(history.history['val_accuracy'], label='val accuracy')

    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend()

    # loss plot 
    plt.subplot(1, 2, 2)

    # training loss over epochs
    plt.plot(history.history['loss'], label='train loss')

    # validation loss over epochs
    plt.plot(history.history['val_loss'], label='val loss')

    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()

    # adjust layout to prevent overlap
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, 'cnn_learning_curves.png'))

    plt.close()


def evaluate_cnn(model, x_test, y_test, classes, results_dir):

    # predict class probabilities for test set
    y_pred_prob = model.predict(x_test, verbose=0)

    # convert probabilities to class indices (argmax)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\nCNN evaluation")

    # generate detailed classification metrics for each class: precision, recall, f1-score
    report = classification_report(
        y_test,
        y_pred,
        target_names=classes,
        zero_division=0,
        output_dict=True
    )

    print(classification_report(
        y_test,
        y_pred,
        target_names=classes,
        zero_division=0
    ))

    # save metrics to shared CSV for report generation
    save_metrics(
        model_name="Custom CNN",
        dataset="custom_aug",
        accuracy=report['accuracy'],
        macro_f1=report['macro avg']['f1-score'],
        macro_precision=report['macro avg']['precision'],
        macro_recall=report['macro avg']['recall'],
        notes="3 conv blocks, dropout=0.5"
    )

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))

    # plot confusion matrix as heatmap
    sns.heatmap(
        cm,
        annot=True,          # show values inside cells
        fmt='d',             # integer format
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title('Confusion matrix - CNN')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(os.path.join(results_dir, 'cm_cnn.png'))

    plt.close()


# cnn architecture builder
def build_cnn(input_shape, num_classes):

    # sequential model: layers stacked one after another
    model = Sequential([

        # input layer defines expected image shape
        Input(shape=input_shape),


        # first convolutional block

        # convolution extracts low-level features (edges, textures)
        Conv2D(32, (3, 3), activation='relu'),

        # pooling reduces spatial dimensions and computation cost
        MaxPooling2D((2, 2)),

        # second convolutional block

        # deeper feature extraction
        Conv2D(64, (3, 3), activation='relu'),

        # downsampling again
        MaxPooling2D((2, 2)),

        # third convolutional block

        # even higher-level feature extraction
        Conv2D(128, (3, 3), activation='relu'),

        # final spatial reduction
        MaxPooling2D((2, 2)),

        # fully connected classifier

        # flatten 3d feature maps into 1d vector
        Flatten(),

        # dense layer for learning nonlinear combinations of features
        Dense(128, activation='relu'),

        # dropout reduces overfitting by randomly disabling neurons
        Dropout(0.5),

        # output layer:
        # one neuron per emotion class
        # (softmax converts outputs to probability distribution)
        Dense(num_classes, activation='softmax')
    ])

    # compile model configuration
    model.compile(
        optimizer=Adam(learning_rate=0.001),

        # sparse categorical crossentropy used for integer labels
        loss='sparse_categorical_crossentropy',

        # accuracy metric for evaluation
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":

    data_dir = "../data/processed/arrays"
    results_dir = "../results"

    os.makedirs(results_dir, exist_ok=True)

    # load image dataset (features)
    x_data = np.load(os.path.join(data_dir, "x_custom_aug.npy"))

    # load labels (emotion classes)
    y_data = np.load(os.path.join(data_dir, "y_custom_aug.npy"))

    # load class names from file
    with open(os.path.join(data_dir, "classes.txt"), "r") as f:
        classes = f.read().split(",")


    # split dataset into training and testing sets
    # 80% training, 20% testing
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=42
    )

    print("Building and training CNN...")

    # create cnn model based on input shape and number of classes
    model = build_cnn(x_data.shape[1:], len(classes))

    # model training

    history = model.fit(
        x_train,
        y_train,

        epochs=30,

        # number of samples per gradient update
        batch_size=16,

        # reserve part of training data for validation
        validation_split=0.1,

        # show training progress
        verbose=1
    )

    plot_learning_curves(history, results_dir)

    evaluate_cnn(model, x_test, y_test, classes, results_dir)

    print(f"\nCNN training finished. Plots saved in {results_dir}.")