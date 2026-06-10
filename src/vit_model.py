import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# shared utility for saving metrics across all scripts
from save_metrics import save_metrics

# keras functional API components
from tensorflow.keras import layers, models
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

    plt.title('Vision Transformer (Vit) model - accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend()

    # loss subplot
    plt.subplot(1, 2, 2)

    # training loss over epochs
    plt.plot(history.history['loss'], label='train loss')

    # validation loss over epochs
    plt.plot(history.history['val_loss'], label='val loss')

    plt.title('Vision Transformer (Vit) model - loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()

    # adjust layout to avoid overlap
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, 'vit_learning_curves.png'))

    plt.close()

def evaluate_vit(model, x_test, y_test, classes, results_dir):

    # predict class probabilities
    y_pred_prob = model.predict(x_test, verbose=0)

    # convert probabilities to predicted class index
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\nVision Transformer (Vit) evaluation")

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
        model_name="Vision Transformer (ViT)",
        dataset="custom_aug",
        accuracy=report['accuracy'],
        macro_f1=report['macro avg']['f1-score'],
        macro_precision=report['macro avg']['precision'],
        macro_recall=report['macro avg']['recall'],
        notes="custom mini-ViT, 2 transformer blocks, patch_size=8"
    )

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))

    # plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,          # show values inside cells
        fmt='d',             # integer format
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title('Confusion matrix - Vision Transformer')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(os.path.join(results_dir, 'cm_vit.png'))

    plt.close()


# mlp block used inside transformer
def mlp(x, hidden_units, dropout_rate):

    # simple feed-forward network used inside transformer blocks
    for units in hidden_units:

        # dense layer with gelu activation (common in transformers)
        x = layers.Dense(units, activation='gelu')(x)

        # dropout for regularization
        x = layers.Dropout(dropout_rate)(x)

    return x


# vision transformer builder
def build_vit_model(input_shape, num_classes):

    # hyperparameters

    # size of each image patch (8x8 pixels)
    patch_size = 8

    # number of patches per image
    # (image_height / patch_size) * (image_width / patch_size)
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    # embedding dimension for each patch
    projection_dim = 64

    # number of attention heads in multi-head attention
    num_heads = 4

    # number of transformer encoder blocks
    transformer_layers = 2

    # input layer
    inputs = layers.Input(shape=input_shape)

    # 1. patch extraction + embedding

    # convert image into patch embeddings using convolution
    # each patch is projected into a vector of size projection_dim
    patches = layers.Conv2D(
        projection_dim,
        kernel_size=patch_size,
        strides=patch_size
    )(inputs)

    # reshape patch grid into sequence of patch tokens
    patch_seq = layers.Reshape((num_patches, projection_dim))(patches)

    # 2. positional encoding

    # create position indices for patches
    positions = tf.range(start=0, limit=num_patches, delta=1)

    # learnable positional embeddings
    pos_emb = layers.Embedding(
        input_dim=num_patches,
        output_dim=projection_dim
    )(positions)

    # add positional information to patch embeddings
    encoded_patches = patch_seq + pos_emb

    # 3. transformer encoder blocks

    x = encoded_patches

    for _ in range(transformer_layers):

        # layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)

        # multi-head self attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=0.1
        )(x1, x1)

        # residual connection (skip connection)
        x2 = layers.Add()([attention_output, x])

        # layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # feed-forward network (mlp block)
        x3 = mlp(
            x3,
            hidden_units=[128, projection_dim],
            dropout_rate=0.1
        )

        # second residual connection
        x = layers.Add()([x3, x2])

    # 4. classification head

    # normalize final transformer output
    representation = layers.LayerNormalization(epsilon=1e-6)(x)

    # aggregate information across all patches
    representation = layers.GlobalAveragePooling1D()(representation)

    # dropout for regularization
    representation = layers.Dropout(0.3)(representation)

    # final classification layer
    outputs = layers.Dense(num_classes, activation="softmax")(representation)

    # build model
    model = models.Model(inputs=inputs, outputs=outputs)

    # compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":

    print("Starting vision transformer setup...")

    data_dir = "../data/processed/arrays"

    results_dir = "../results"

    os.makedirs(results_dir, exist_ok=True)

    # load dataset

    # load image data (grayscale)
    x_data = np.load(os.path.join(data_dir, "x_custom_aug.npy"))

    # load labels
    y_data = np.load(os.path.join(data_dir, "y_custom_aug.npy"))

    # load class names
    with open(os.path.join(data_dir, "classes.txt"), "r") as f:
        classes = f.read().split(",")

    # split dataset into training and testing subsets
    # 80% training, 20% testing
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=42
    )

    # model preparation

    print("Building and training custom mini-vit model...")

    # define input shape
    input_shape = x_data.shape[1:]

    # number of classes
    num_classes = len(classes)

    # build vision transformer model
    vit = build_vit_model(input_shape, num_classes)

    # training phase

    # note: transformers usually need more epochs to converge
    history = vit.fit(
        x_train,
        y_train,
        epochs=40,
        batch_size=16,
        validation_split=0.1,
        verbose=1
    )

    plot_learning_curves(history, results_dir)

    evaluate_vit(vit, x_test, y_test, classes, results_dir)

    print(f"\nVision Transformer training finished. Check {results_dir} for plots.")