import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# train-test split utility from sklearn
from sklearn.model_selection import train_test_split

# evaluation metrics for classification tasks
from sklearn.metrics import classification_report, confusion_matrix

# keras components for building neural networks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


def evaluate_nn_model(model, x_test, y_test, classes, model_name, results_dir):

    # run prediction on test dataset
    y_pred_prob = model.predict(x_test, verbose=0)

    # convert probability distribution to class index
    # argmax selects class with highest probability
    y_pred = np.argmax(y_pred_prob, axis=1)

    print(f"\n{model_name} evaluation")

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

    # plot confusion matrix as heatmap
    sns.heatmap(
        cm,
        annot=True,          # display numeric values in cells
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


# mlp model builder
def create_mlp(input_shape, num_classes, hidden_layers, activation='relu', dropout_rate=0.0):

    # initialize sequential neural network model
    model = Sequential()

    # flatten input image into 1d vector
    # (required because dense layers expect flat input)
    model.add(Flatten(input_shape=input_shape))

    # dynamically create hidden layers based on configuration
    for units in hidden_layers:

        # fully connected layer with specified number of neurons
        model.add(Dense(units, activation=activation))

        # optional dropout layer to reduce overfitting (randomly disables neurons during training)
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # output layer:
    # number of neurons = number of emotion classes
    # (softmax converts outputs into probability distribution)
    model.add(Dense(num_classes, activation='softmax'))

    # compile model:
    # optimizer: adam (adaptive learning rate optimization)
    # loss: sparse categorical crossentropy (for integer labels)
    # metric: accuracy
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":

    print("Starting mlp models training...")

    data_dir = "../data/processed/arrays"

    results_dir = "../results"

    os.makedirs(results_dir, exist_ok=True)

    # load image dataset (features)
    x_data = np.load(os.path.join(data_dir, "x_custom_aug.npy"))

    # load corresponding labels (emotion classes)
    y_data = np.load(os.path.join(data_dir, "y_custom_aug.npy"))

    # load class names from file
    with open(os.path.join(data_dir, "classes.txt"), "r") as f:
        classes = f.read().split(",")

    # number of emotion classes
    num_classes = len(classes)

    # input shape of one sample (height, width, channels)
    input_shape = x_data.shape[1:]

    # split dataset into training and testing subsets
    # 80% training, 20% testing
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=42  # ensures reproducible results
    )

    epochs = 15
    batch_size = 16

    # experiment 1: shallow mlp
    print("\nTraining experiment 1: shallow mlp")

    # single hidden layer network
    mlp_1 = create_mlp(
        input_shape,
        num_classes,
        hidden_layers=[128],
        activation='relu'
    )

    # train model
    mlp_1.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    evaluate_nn_model(mlp_1, x_test, y_test, classes, "MLP Shallow", results_dir)

    # experiment 2: deep mlp + dropout
    print("\nTraining experiment 2: deep mlp with dropout")

    # deeper architecture with regularization
    mlp_2 = create_mlp(
        input_shape,
        num_classes,
        hidden_layers=[256, 128],
        activation='relu',
        dropout_rate=0.3
    )

    mlp_2.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    evaluate_nn_model(mlp_2, x_test, y_test, classes, "MLP Deep", results_dir)

    # experiment 3: deep mlp with tanh
    print("\nTraining experiment 3: deep mlp with tanh activation")

    # same architecture but different activation function
    mlp_3 = create_mlp(
        input_shape,
        num_classes,
        hidden_layers=[256, 128],
        activation='tanh',
        dropout_rate=0.3
    )

    mlp_3.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    evaluate_nn_model(mlp_3, x_test, y_test, classes, "MLP Tanh", results_dir)

    print(f"\nAll mlp experiments completed. Check {results_dir} for confusion matrices.")