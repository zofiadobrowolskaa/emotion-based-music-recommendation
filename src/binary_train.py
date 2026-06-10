import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence, to_categorical

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from save_metrics import save_metrics

# valence scheme (positive vs negative)

# instead of  regrouping a 7-class model's predictions post-hoc, MobileNetV2 is trained directly 
# on the 2-class valence labels so the binary objective is optimised end-to-end.

# binary group order: index 0 = positive, index 1 = negative
GROUP_NAMES = ["positive", "negative"]

# emotion -> binary group id
EMOTION_TO_BINARY = {
    "happy": 0, "surprise": 0, "neutral": 0,
    "angry": 1, "disgust": 1, "fear": 1, "sad": 1,
}


class BinaryLabelWrapper(Sequence):
    # wraps a 7-class flow_from_directory iterator and remaps its
    # one-hot labels to the 2-class valence labels on the fly.

    def __init__(self, base_iterator, index_to_binary):
        # base_iterator: the original 7-class DirectoryIterator
        # index_to_binary: array mapping original class index -> 0/1
        self.base = base_iterator
        self.index_to_binary = index_to_binary

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x_batch, y_batch = self.base[idx]

        # original 7-class index for each sample in the batch
        original_idx = np.argmax(y_batch, axis=1)

        # remap to binary id, then back to one-hot (2 columns)
        binary_idx = self.index_to_binary[original_idx]
        y_binary = to_categorical(binary_idx, num_classes=2)

        return x_batch, y_binary

    def on_epoch_end(self):
        # delegate shuffling to the underlying iterator
        self.base.on_epoch_end()


def plot_learning_curves(history, results_dir):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Binary Valence Model (MobileNetV2) - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Binary Valence Model (MobileNetV2) - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'binary_model_learning_curves.png'), dpi=150)
    plt.close()
    print(f"Learning curves saved to {results_dir}/binary_model_learning_curves.png")


def evaluate_binary_model(model, validation_wrapper, class_names, results_dir):

    print("\nEvaluating binary valence model on validation set...")

    y_true = []
    y_pred = []

    for i in range(len(validation_wrapper)):
        x_batch, y_batch = validation_wrapper[i]
        preds = model.predict(x_batch, verbose=0)

        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(y_batch, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nBinary valence model - classification report:")
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    save_metrics(
        model_name="MobileNetV2 (Binary Valence, FER-2013)",
        dataset="FER-2013",
        accuracy=report['accuracy'],
        macro_f1=report['macro avg']['f1-score'],
        macro_precision=report['macro avg']['precision'],
        macro_recall=report['macro avg']['recall'],
        notes="trained directly on 2-class valence labels (pos/neg); frozen backbone, class weights"
    )

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Confusion Matrix - Binary Valence Model (MobileNetV2)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cm_binary_model.png'), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {results_dir}/cm_binary_model.png")


def build_binary_model(input_shape, num_classes):
    # identical architecture to final_train.py, only the head size differs
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    # freeze the convolutional backbone (transfer learning)
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        # 2 output neurons: positive / negative valence
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":

    train_dir = "../data/fer2013/train"
    test_dir = "../data/fer2013/test"

    model_save_path = "../models/binary_emotion_model.h5"
    results_dir = "../results"

    os.makedirs("../models", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # same augmentation policy as the production model
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    # load images as 7-class iterators first (folders are per-emotion)
    train_base = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='rgb',
        batch_size=64,
        class_mode='categorical'
    )

    validation_base = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        color_mode='rgb',
        batch_size=64,
        class_mode='categorical',
        shuffle=False
    )

    # original 7 emotion names ordered by their integer index
    emotion_names = [
        name for name, _ in
        sorted(train_base.class_indices.items(), key=lambda x: x[1])
    ]
    print(f"Detected emotions: {emotion_names}")

    # array mapping original class index -> binary valence id
    index_to_binary = np.array([EMOTION_TO_BINARY[name] for name in emotion_names])
    print(f"Index -> binary map: {dict(enumerate(index_to_binary))}")

    # wrap both iterators so they emit 2-class labels
    train_generator = BinaryLabelWrapper(train_base, index_to_binary)
    validation_generator = BinaryLabelWrapper(validation_base, index_to_binary)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)
    ]

    print("Starting binary valence training on full fer-2013 dataset...")

    model = build_binary_model((48, 48, 3), 2)

    # balanced class weights computed on the binary labels of the whole train set
    binary_train_labels = index_to_binary[train_base.classes]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(binary_train_labels),
        y=binary_train_labels
    )
    weight_dict = dict(enumerate(class_weights))
    print(f"applied class weights: {weight_dict}")

    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=weight_dict,
        verbose=1
    )

    plot_learning_curves(history, results_dir)
    evaluate_binary_model(model, validation_generator, GROUP_NAMES, results_dir)

    print(f"Binary valence model saved successfully at: {model_save_path}")
