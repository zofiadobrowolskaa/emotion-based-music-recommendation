import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

from save_metrics import save_metrics


# aggregation schemes

# each scheme collapses the 7 original emotions into 2 groups.
# the goal is to show that most of the 7-class error comes from
# confusing semantically similar emotions (fear/sad/angry), and
# that the model is reliable at the coarser, valence level that
# actually drives the music recommendation.

# format: scheme_name -> {group_label: [original emotions]}
SCHEMES = {
    "valence_pos_neg": {
        "positive": ["happy", "surprise", "neutral"],
        "negative": ["angry", "disgust", "fear", "sad"],
    },

    "happy_vs_rest": {
        "happy": ["happy"],
        "rest": ["angry", "disgust", "fear", "neutral", "sad", "surprise"],
    },

    # high vs low arousal
    # this is genuinely different from valence and maps directly to
    # the music "energy" attribute used by the recommendation rules
    "arousal_high_low": {
        "high_arousal": ["angry", "fear", "happy", "surprise"],
        "low_arousal": ["disgust", "neutral", "sad"],
    },
}


def build_test_generator(test_dir):
    # recreate the exact validation pipeline from final_train.py
    # (only rescale, no augmentation, no shuffle for stable order)
    test_datagen = ImageDataGenerator(rescale=1./255)

    generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        color_mode='rgb',
        batch_size=64,
        class_mode='categorical',
        shuffle=False
    )

    return generator


def collect_predictions(model, generator):
    # run the 7-class model over the whole test set and gather
    # true labels and predicted labels as integer class indices
    y_true = []
    y_pred = []

    generator.reset()

    for _ in range(len(generator)):
        x_batch, y_batch = next(generator)

        preds = model.predict(x_batch, verbose=0)

        # argmax over softmax outputs gives the predicted class index
        y_pred.extend(np.argmax(preds, axis=1))

        # one-hot labels from flow_from_directory also need argmax
        y_true.extend(np.argmax(y_batch, axis=1))

    return np.array(y_true), np.array(y_pred)


def build_label_map(scheme, class_names):
    # build a lookup: original class index -> aggregated group label
    
    # also return the ordered list of group labels so the confusion
    # matrix axes stay consistent across true/pred
    emotion_to_group = {}
    for group_label, emotions in scheme.items():
        for emotion in emotions:
            emotion_to_group[emotion] = group_label

    # map each original class index to its group label string
    index_to_group = {
        idx: emotion_to_group[name]
        for idx, name in enumerate(class_names)
    }

    # fixed order of the two groups (as declared in the scheme dict)
    group_order = list(scheme.keys())

    return index_to_group, group_order


def evaluate_scheme(scheme_name, scheme, y_true, y_pred, class_names, results_dir):
    # aggregate the 7-class predictions into the 2-class scheme and
    # recompute accuracy, macro-f1 and balanced accuracy
    index_to_group, group_order = build_label_map(scheme, class_names)

    # turn group labels into integer ids in the declared order
    group_to_id = {label: i for i, label in enumerate(group_order)}

    # remap both true and predicted labels to the aggregated ids
    y_true_agg = np.array([group_to_id[index_to_group[i]] for i in y_true])
    y_pred_agg = np.array([group_to_id[index_to_group[i]] for i in y_pred])

    # core metrics
    acc = accuracy_score(y_true_agg, y_pred_agg)

    # macro-f1 and balanced accuracy are the honest metrics here:
    # plain accuracy is inflated when one group is the majority
    macro_f1 = f1_score(y_true_agg, y_pred_agg, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true_agg, y_pred_agg)

    report = classification_report(
        y_true_agg,
        y_pred_agg,
        target_names=group_order,
        zero_division=0,
        output_dict=True
    )

    print(f"\n=== scheme: {scheme_name} ===")
    print(f"  groups: {scheme}")
    print(classification_report(
        y_true_agg,
        y_pred_agg,
        target_names=group_order,
        zero_division=0
    ))
    print(f"  accuracy={acc:.4f}  macro_f1={macro_f1:.4f}  balanced_acc={bal_acc:.4f}")

    cm = confusion_matrix(y_true_agg, y_pred_agg)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=group_order,
        yticklabels=group_order
    )
    plt.title(f'Aggregated Confusion Matrix - {scheme_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    cm_path = os.path.join(results_dir, f'cm_aggregation_{scheme_name}.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"confusion matrix saved to {cm_path}")

    save_metrics(
        model_name=f"MobileNetV2 aggregated ({scheme_name})",
        dataset="FER-2013",
        accuracy=acc,
        macro_f1=macro_f1,
        macro_precision=report['macro avg']['precision'],
        macro_recall=report['macro avg']['recall'],
        notes=f"post-hoc 2-class aggregation of 7-class model; balanced_acc={bal_acc:.4f}"
    )

    return {
        'scheme': scheme_name,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'balanced_acc': bal_acc,
    }


if __name__ == "__main__":

    test_dir = "../data/fer2013/test"
    model_path = "../models/final_emotion_model.h5"
    results_dir = "../results"

    os.makedirs(results_dir, exist_ok=True)

    print("Loading trained 7-class model...")
    model = load_model(model_path)

    print("Building test generator...")
    generator = build_test_generator(test_dir)

    # class names ordered by their integer index
    class_names = [
        name for name, _ in
        sorted(generator.class_indices.items(), key=lambda x: x[1])
    ]
    print(f"Detected classes: {class_names}")

    print("Collecting 7-class predictions over the test set...")
    y_true, y_pred = collect_predictions(model, generator)

    # reference: the original 7-class accuracy (for the comparison table)
    base_acc = accuracy_score(y_true, y_pred)
    print(f"\nBaseline 7-class accuracy: {base_acc:.4f}")

    # evaluate every aggregation scheme on the same predictions
    summary = [{
        'scheme': '7-class (baseline)',
        'accuracy': base_acc,
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'balanced_acc': balanced_accuracy_score(y_true, y_pred),
    }]

    for scheme_name, scheme in SCHEMES.items():
        summary.append(
            evaluate_scheme(scheme_name, scheme, y_true, y_pred, class_names, results_dir)
        )

    # final comparison table printed to console
    print(f"Aggregation comparison (post-hoc, same model)")
    print(f"{'scheme':<24}{'accuracy':>10}{'macro_f1':>10}{'bal_acc':>10}")
    for row in summary:
        print(f"{row['scheme']:<24}{row['accuracy']:>10.4f}"
              f"{row['macro_f1']:>10.4f}{row['balanced_acc']:>10.4f}")

    print(f"\n2-class confusion matrices and metrics saved to {results_dir}.")
