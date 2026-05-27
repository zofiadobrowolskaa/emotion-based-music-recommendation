import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# train-test split utility from sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# simple baseline models used for comparison
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# naive bayes for gaussian-distributed continuous features (normalized pixel values)
from sklearn.naive_bayes import GaussianNB

# evaluation metrics for classification tasks
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(y_true, y_pred, classes, model_name, results_dir):

    print(f"\n{model_name} evaluation")

    # generate detailed classification metrics for each class: precision, recall, f1-score
    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        zero_division=0,
        output_dict=True
    )

    print(classification_report(
        y_true,
        y_pred,
        target_names=classes,
        zero_division=0
    ))

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
    plt.tight_layout()

    plot_path = os.path.join(
        results_dir,
        f'cm_{model_name.replace(" ", "_").lower()}.png'
    )

    plt.savefig(plot_path, dpi=150)

    plt.close()

    # return macro-averaged f1 for summary table
    return report['macro avg']['f1-score'], report['accuracy']


def run_cross_validation(model, x_flat, y_data, model_name, cv_folds=5):

    # stratified k-fold preserves class proportions in every fold
    # this is critical when classes are imbalanced (e.g. more "happy" than "fear")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # compute accuracy across all folds
    scores = cross_val_score(
        model,
        x_flat,
        y_data,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1  # use all available cpu cores
    )

    print(f"\n{model_name} - {cv_folds}-fold cross-validation:")
    print(f"  fold accuracies: {[f'{s:.3f}' for s in scores]}")
    print(f"  mean accuracy:   {scores.mean():.4f} (+/- {scores.std():.4f})")

    return scores.mean(), scores.std()


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
    # required for classical ML models (kNN, decision tree, naive bayes)
    n_samples = x_data.shape[0]

    # reshape: (samples, 48, 48, 1) -> (samples, 2304)
    x_flattened = x_data.reshape((n_samples, -1))

    print(f"Data flattened. new shape: {x_flattened.shape}")

    # split dataset into training and testing subsets
    # 80% training, 20% testing, stratified to preserve class balance
    x_train, x_test, y_train, y_test = train_test_split(
        x_flattened,
        y_data,
        test_size=0.2,
        random_state=42,      # ensures reproducibility of results
        stratify=y_data       # preserve class distribution in both splits
    )

    # dictionary to store summary results for all models
    summary_rows = []

    # ==============================
    # model 1: k-nearest neighbors
    # ==============================

    print("\nTraining k-nearest neighbors...")

    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(x_train, y_train)

    y_pred_knn = knn.predict(x_test)

    f1_knn, acc_knn = evaluate_model(y_test, y_pred_knn, classes, "kNN", results_dir)

    # cross-validate on the entire (non-split) dataset for more reliable estimate
    cv_mean_knn, cv_std_knn = run_cross_validation(knn, x_flattened, y_data, "kNN")

    summary_rows.append({
        'Model': 'kNN (k=3)',
        'Test Accuracy': round(acc_knn, 4),
        'Macro F1': round(f1_knn, 4),
        'CV Mean Accuracy': round(cv_mean_knn, 4),
        'CV Std': round(cv_std_knn, 4)
    })

    # ==============================
    # model 2: decision tree
    # ==============================

    print("\nTraining decision tree...")

    dt = DecisionTreeClassifier(random_state=42)

    dt.fit(x_train, y_train)

    y_pred_dt = dt.predict(x_test)

    f1_dt, acc_dt = evaluate_model(y_test, y_pred_dt, classes, "Decision Tree", results_dir)

    cv_mean_dt, cv_std_dt = run_cross_validation(dt, x_flattened, y_data, "Decision Tree")

    summary_rows.append({
        'Model': 'Decision Tree',
        'Test Accuracy': round(acc_dt, 4),
        'Macro F1': round(f1_dt, 4),
        'CV Mean Accuracy': round(cv_mean_dt, 4),
        'CV Std': round(cv_std_dt, 4)
    })

    # ==============================
    # model 3: naive bayes
    # ==============================

    print("\nTraining Naive Bayes...")

    # gaussian naive bayes assumes each feature follows a normal distribution
    # this is a reasonable assumption for normalized pixel intensities [0, 1]
    nb = GaussianNB()

    nb.fit(x_train, y_train)

    y_pred_nb = nb.predict(x_test)

    f1_nb, acc_nb = evaluate_model(y_test, y_pred_nb, classes, "Naive Bayes", results_dir)

    cv_mean_nb, cv_std_nb = run_cross_validation(nb, x_flattened, y_data, "Naive Bayes")

    summary_rows.append({
        'Model': 'Naive Bayes',
        'Test Accuracy': round(acc_nb, 4),
        'Macro F1': round(f1_nb, 4),
        'CV Mean Accuracy': round(cv_mean_nb, 4),
        'CV Std': round(cv_std_nb, 4)
    })

    # ==============================
    # save summary table to csv
    # ==============================

    # create summary dataframe with all results
    summary_df = pd.DataFrame(summary_rows)

    csv_path = os.path.join(results_dir, "baseline_results_summary.csv")

    summary_df.to_csv(csv_path, index=False)

    print(f"\nBaseline model comparison:")
    print(summary_df.to_string(index=False))
    print(f"\nEvaluation completed. Confusion matrices and summary saved to {results_dir}.")
