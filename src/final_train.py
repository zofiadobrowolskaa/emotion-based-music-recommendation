import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import pre-trained mobilenetv2 architecture
from tensorflow.keras.applications import MobileNetV2

# sequential api for stacking layers
from tensorflow.keras.models import Sequential

# layers used for custom classification head
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout)

# optimizer for training
from tensorflow.keras.optimizers import Adam

# image preprocessing and augmentation tools
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# callbacks for better training control
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


def plot_learning_curves(history, results_dir):
    # plot accuracy and loss curves for the final production model

    plt.figure(figsize=(12, 4))

    # accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Final Model (MobileNetV2) - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Final Model (MobileNetV2) - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'final_model_learning_curves.png'), dpi=150)
    plt.close()

    print(f"Learning curves saved to {results_dir}/final_model_learning_curves.png")


def evaluate_final_model(model, validation_generator, class_names, results_dir):
    # run full evaluation on the validation set and save confusion matrix + metrics

    print("\nEvaluating final model on validation set...")

    # gather all predictions and true labels from the generator
    y_true = []
    y_pred = []

    # reset generator to start from the beginning
    validation_generator.reset()

    for i in range(len(validation_generator)):
        x_batch, y_batch = next(validation_generator)

        # predict class probabilities for this batch
        preds = model.predict(x_batch, verbose=0)

        # argmax gives the predicted class index
        y_pred.extend(np.argmax(preds, axis=1))

        # argmax also needed for one-hot encoded labels from flow_from_directory
        y_true.extend(np.argmax(y_batch, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nFinal model - classification report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    ))

    # build confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title('Confusion Matrix - Final Model (MobileNetV2 Transfer Learning)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, 'cm_final_model.png'), dpi=150)
    plt.close()

    print(f"Confusion matrix saved to {results_dir}/cm_final_model.png")


def build_final_model(input_shape, num_classes):

    # load pre-trained mobilenetv2 with weights from imagenet

    # include_top=False removes original imagenet classifier
    # (this allows us to attach our own emotion classifier)
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    # freeze convolutional backbone (base layers)
    # pre-trained weights remain unchanged during training
    # (this preserves learned visual features)
    base_model.trainable = False

    # build custom classification head
    model = Sequential([

        base_model,  # feature extractor

        # converts feature maps into compact vector
        # much lighter than flattening entire tensor
        GlobalAveragePooling2D(),

        # dense layer learns emotion-specific patterns
        Dense(256, activation='relu'),

        # dropout randomly disables neurons during training, helps prevent overfitting
        Dropout(0.4),

        # final classification layer, one neuron per emotion class
        Dense(num_classes, activation='softmax')
    ])

    model.compile(

        # lower learning rate improves stability
        optimizer=Adam(learning_rate=0.0005),

        # categorical crossentropy used because labels
        # are one-hot encoded by flow_from_directory
        loss='categorical_crossentropy',

        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":

    train_dir = "../data/fer2013/train"
    test_dir = "../data/fer2013/test"

    model_save_path = "../models/final_emotion_model.h5"
    results_dir = "../results"

    os.makedirs("../models", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # training data augmentation

    # augmentation artificially increases dataset diversity
    # (this improves model generalization ability)

    train_datagen = ImageDataGenerator(

        # normalize pixel values to range [0, 1]
        rescale=1./255,

        # random image rotations
        rotation_range=20,

        # random zoom operations
        zoom_range=0.2,

        # random horizontal flips
        horizontal_flip=True
    )

    # validation preprocessing

    # validation data should remain unchanged, only normalization is applied
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    # load training images

    # flow_from_directory automatically:
    # - reads images from folders
    # - assigns labels based on folder names
    # - creates batches during training

    train_generator = train_datagen.flow_from_directory(

        # dataset location
        train_dir,

        # resize all images to 48x48
        target_size=(48, 48),

        # mobilenetv2 requires rgb input
        # even though fer images are grayscale
        color_mode='rgb',

        batch_size=64,

        # one-hot encoded labels
        class_mode='categorical'
    )

    # load validation images

    validation_generator = test_datagen.flow_from_directory(

        # validation dataset location
        test_dir,

        # resize images to same dimensions
        target_size=(48, 48),

        # rgb conversion
        color_mode='rgb',

        batch_size=64,

        # one-hot encoded labels
        class_mode='categorical',

        # disable shuffling for consistent evaluation order
        shuffle=False
    )

    # extract class names sorted by their integer index (as assigned by flow_from_directory)
    class_names = [
        name for name, _ in
        sorted(train_generator.class_indices.items(), key=lambda x: x[1])
    ]

    print(f"Detected classes: {class_names}")

    # callbacks configuration

    # callbacks automate important training behaviors

    callbacks = [

        # stop training if validation loss
        # does not improve for 5 epochs

        # restore_best_weights=True reloads best-performing model automatically
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),

        # save only best-performing model based on validation accuracy
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

    print("Starting final training on full fer-2013 dataset...")

    # create final transfer learning model
    model = build_final_model(
        (48, 48, 3),
        7
    )

    # =========================
    # compute class weights
    # =========================

    # calculate balanced class weights based on
    # the number of samples in each emotion class
    #
    # classes with fewer images receive higher weights,
    # which helps reduce dataset imbalance during training
    #
    # example:
    # if "happy" has many samples and "fear" has few,
    # the model will pay more attention to "fear"
    class_weights = compute_class_weight(

        # automatically compute inverse-frequency weights
        class_weight='balanced',

        # list of unique class labels
        classes=np.unique(train_generator.classes),

        # actual class labels for all training images
        y=train_generator.classes
    )

    # convert weights array into dictionary format
    # required by tensorflow model.fit(class_weight=...)
    #
    # example output:
    # {0: 1.2, 1: 0.8, 2: 1.5, ...}
    weight_dict = dict(enumerate(class_weights))

    # display computed weights in console
    print(f"applied class weights: {weight_dict}")

    # train model

    history = model.fit(

        # training batches
        train_generator,

        epochs=50,

        # validation dataset
        validation_data=validation_generator,

        # training callbacks
        callbacks=callbacks,

        class_weight=weight_dict,

        # display detailed training logs
        verbose=1
    )

    # save learning curves for the report
    plot_learning_curves(history, results_dir)

    # evaluate and save confusion matrix
    evaluate_final_model(model, validation_generator, class_names, results_dir)

    print(f"Final model saved successfully at: {model_save_path}")
