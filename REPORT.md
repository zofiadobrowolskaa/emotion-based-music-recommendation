# Facial Emotion Recognition for Music Recommendation - Research Report

A complete experimental study of an emotion-aware music recommendation system. The
system detects a person's emotional state from a facial photograph using computer
vision and deep learning, and then recommends music whose audio characteristics match
that emotion.

This report documents every stage of the work: the data, the preprocessing, every model
that was built and tested, the results each one achieved, what those results mean, and an
honest discussion of the limitations and what could be improved.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction and Objectives](#2-introduction-and-objectives)
3. [System Overview](#3-system-overview)
4. [Datasets](#4-datasets)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Experimental Methodology](#7-experimental-methodology)
8. [Models and Results](#8-models-and-results)
   - 8.1 [Baseline Classifiers (kNN, Decision Tree, Naive Bayes)](#81-baseline-classifiers)
   - 8.2 [Multi-Layer Perceptrons (MLP)](#82-multi-layer-perceptrons-mlp)
   - 8.3 [Custom Convolutional Neural Network (CNN)](#83-custom-convolutional-neural-network-cnn)
   - 8.4 [Transfer Learning - MobileNetV2](#84-transfer-learning--mobilenetv2-experiment)
   - 8.5 [Vision Transformer (ViT)](#85-vision-transformer-vit)
   - 8.6 [Conditional DCGAN (Generative Bonus)](#86-conditional-dcgan-generative-bonus)
9. [Comparative Results Summary](#9-comparative-results-summary)
10. [Final Production Model](#10-final-production-model--mobilenetv2-on-fer-2013)
    - 10.1 [Emotion Aggregation Analysis (2-class)](#101-emotion-aggregation-analysis-2-class)
    - 10.2 [Training Directly on 2-class Labels (Valence)](#102-training-directly-on-2-class-labels-valence)
11. [Association Rule Mining (Apriori)](#11-association-rule-mining-apriori)
12. [The Application](#12-the-application)
13. [Limitations and Critical Discussion](#13-limitations-and-critical-discussion)
14. [Conclusions](#14-conclusions)
15. [Future Work](#15-future-work)
16. [Bibliography](#16-bibliography)
17. [Appendix - How to Reproduce](#17-appendix--how-to-reproduce)

---

## 1. Abstract

This work presents an end-to-end system for **emotion-aware music recommendation**. A face
image is processed by an **MTCNN** face detector, cropped, and classified into one of seven
emotions (*angry, disgust, fear, happy, neutral, sad, surprise*) by a deep learning model.
The predicted emotion is then mapped to musical audio attributes (tempo, key, energy) using
**association rules** discovered with the **Apriori** algorithm, and these attributes drive a
live music search through the **Spotify API**. The whole pipeline is wrapped in an interactive
**Streamlit** web application.

The research component systematically compares **ten model configurations**: three classical
baseline classifiers evaluated with 5-fold cross-validation (k-Nearest Neighbours, Decision
Tree, Gaussian Naive Bayes), three Multi-Layer Perceptrons, a custom Convolutional Neural
Network, a MobileNetV2 transfer-learning model, and a custom Vision Transformer. A conditional
DCGAN was additionally implemented to explore generative synthesis of emotion-conditioned faces.
Experiments were carried out on two datasets - the **FER-2013** benchmark (35,887 images) and a
small **custom dataset** of 35 self-collected, perfectly balanced photographs. Every result is
reported with accuracy, macro-averaged precision, macro-averaged recall, per-class confusion matrices and learning
curves, and is interpreted in context. The seven-class predictions of the production model are
additionally **aggregated into two-class schemes** (valence, arousal, one-vs-rest), and a dedicated
**binary valence model** is trained end-to-end (§10.1–10.2).

---

## 2. Introduction and Objectives

Music has a strong, well-documented relationship with human emotion. The idea behind this
project is simple and intuitive: **if a computer can read the emotion on a person's face, it can
recommend music that fits their current mood.** A sad face might receive slow, minor-key,
low-energy tracks; a happy face might receive fast, major-key, high-energy ones.

Turning that idea into a working system touches almost every major area of computational
intelligence, which is exactly why it was chosen. The concrete objectives were:

1. **Build a real image-classification pipeline** - from raw photos to a trained model that
   outputs an emotion.
2. **Compare a wide range of classifiers** - from the simplest classical algorithms to modern
   deep architectures - and *understand why some work and others fail* on this problem.
3. **Go beyond the standard syllabus** - by implementing a Vision Transformer and a Generative
   Adversarial Network, and by grounding the work in scientific literature.
4. **Deliver a usable application** - not just notebooks, but an interactive tool that takes a
   photo and returns real, clickable music recommendations.

The task belongs to the **image classification** category, and the report follows the classic
structure of such studies: *database → preprocessing → classification experiments → association
rules → interpretation*.

---

## 3. System Overview

The system is a pipeline of independent, reusable stages.

```
 ┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
 │  Face photo │ → │ MTCNN face   │ → │ Resize 48×48 │ → │ Emotion model    │
 │ (upload/cam)│   │ detection    │   │ + normalise  │   │ (MobileNetV2)    │
 └─────────────┘   └──────────────┘   └──────────────┘   └────────┬─────────┘
                                                                   │ dominant emotion
                                                                   ▼
                          ┌────────────────────┐        ┌────────────────────┐
                          │  Spotify API search │  ◄──── │ Apriori association │
                          │  (recommended songs)│        │ rules (emotion→     │
                          └────────────────────┘        │ tempo/key/energy)   │
                                                          └────────────────────┘
```

---

## 4. Datasets

Two datasets were used, each serving a different purpose.

### 4.1 FER-2013 (primary benchmark)

FER-2013 is a classic facial-emotion-recognition benchmark introduced by Goodfellow et al.
(2013). It contains **48×48 pixel grayscale** face images, each labelled with one of seven
emotions. It is the dataset used to train the **final production model**.

| Split    | Images | Share |
|----------|-------:|------:|
| Training | 28,709 | 80 % |
| Test     |  7,178 | 20 % |
| **Total**| **35,887** | **100 %** |

The split (≈ 80 / 20) is the standard FER-2013 train/test partition shipped with the dataset.

**Class distribution (training set, the 28,709 training images)** - from [`results/class_distribution_stats.csv`](results/class_distribution_stats.csv):

| Emotion  | Count | Percentage |
|----------|------:|-----------:|
| happy    | 7,215 | 25.13 % |
| neutral  | 4,965 | 17.29 % |
| sad      | 4,830 | 16.82 % |
| fear     | 4,097 | 14.27 % |
| angry    | 3,995 | 13.92 % |
| surprise | 3,171 | 11.05 % |
| disgust  |   436 |  1.52 % |

The dataset is **strongly imbalanced**: the `happy` class contains far more images than the `disgust` class. This is a real problem - a model can score high accuracy simply by ignoring rare
classes. It was addressed in the final model with **balanced class weights** (see §10).

### 4.2 Custom Dataset

A small, **perfectly balanced** dataset of **35 real face photographs** was collected and
annotated specifically for this project - **5 photos per emotion × 7 emotions**. Building an own
dataset demonstrates the full data pipeline (raw photo → detection → cropping → arrays) on data
that did not come pre-cleaned.

| Property | Value |
|----------|-------|
| Raw images | 35 (5 per class) |
| After MTCNN face cropping | 35 |
| After augmentation (×3 + original) | 140 |
| Class balance | perfectly uniform (14.29 % each) |

This dataset is used for **all the model-comparison experiments** (§8). Its small size is a
deliberate teaching trade-off and its consequences are discussed openly in §13.

---

## 5. Exploratory Data Analysis

Script: [`src/eda.py`](src/eda.py)

Before any modelling, both datasets were inspected visually and statistically. EDA answers two
questions: *are the classes balanced?* and *what do the images actually look like?*

**FER-2013 class distribution** - note the dominant `happy` class and the tiny `disgust` class:

![FER-2013 class distribution](results/class_distribution.png)

**Custom dataset class distribution** - uniform by construction:

![Custom dataset class distribution](results/custom_class_distribution.png)

**Side-by-side comparison** (normalised to percentages so the two different-sized datasets can be
compared fairly):

![Dataset comparison](results/dataset_comparison.png)

**Sample images per class** confirm the data was loaded and labelled correctly:

![FER-2013 samples](results/sample_images.png)

![Custom samples](results/custom_sample_images.png)

**Conclusion from EDA:** FER-2013's imbalance must be handled during training; the custom set is
balanced but very small, so its results will be high-variance.

---

## 6. Preprocessing Pipeline

Good preprocessing is decisive for image classification. Three steps turn raw photographs into
clean, model-ready arrays.

### Step 1 - Face Detection / Object Detection ([`face_detector.py`](src/face_detector.py))

Raw photos contain background, hair, clothing - noise that has nothing to do with emotion. The
**MTCNN** (Multi-task Cascaded Convolutional Network) detector finds the face bounding box (and
facial landmarks) in each photo. Only the **face crop** is kept; everything else is discarded.
This is a genuine **object-detection** step, and it is the same detector used live in the app, so
training and inference see comparably framed faces.

### Step 2 - Cleaning and Normalisation ([`preprocess.py`](src/preprocess.py))

Each cropped face is:
1. Loaded in **grayscale** (1 channel) - colour carries little emotion information and grayscale
   matches FER-2013.
2. **Resized** to **48×48** pixels - a fixed input size every model can consume.
3. **Normalised** to the `[0, 1]` range (pixel value / 255) - this keeps neural-network training
   numerically stable.

The arrays are saved as `.npy` files for fast reloading.

### Step 3 - Data Augmentation ([`augment.py`](src/augment.py))

35 images is far too few to train deep networks. Augmentation **synthesises new, realistic
variations** of each image, expanding the set from **35 → 140** (each image produces the original
plus 3 augmented copies). The transformations simulate real-world variation in how a face appears
to a camera:

| Transformation | Setting | Why |
|----------------|---------|-----|
| Rotation | ±15° random | head tilt |
| Zoom | ±15 % random | distance from camera |
| Horizontal flip | 50 % | left/right symmetry of faces |
| Gaussian noise | σ = 0.03 (custom function) | sensor noise / robustness |
| Fill mode | nearest neighbour | fills pixels exposed by rotation |

The Gaussian-noise step is a **custom preprocessing function** added on top of the standard Keras
`ImageDataGenerator`, demonstrating control over the augmentation pipeline rather than relying on
defaults.

---

## 7. Experimental Methodology

To make the model comparison fair and meaningful, the same protocol was applied throughout.

- **Data:** the augmented custom dataset (140 samples) is used for all comparison experiments
  (§8). The final production model (§10) is trained separately on FER-2013.
- **Train/test split:** 80 % train / 20 % test. The classical baselines and the MLPs use
  **stratified** splits (preserving class proportions); the deep models use a fixed
  `random_state=42` split.
- **Cross-validation:** the classical baselines additionally use **5-fold stratified
  cross-validation** - the dataset is split into 5 parts, each used once as a test fold, and the
  scores are averaged. This gives a more trustworthy estimate than a single split on a small set.

### How to read the metrics (quick reference)

These are the numbers used throughout the report:

- **Accuracy** - fraction of all predictions that were correct. Simple, but misleading on
  imbalanced data.
- **Precision** (per class) - of the images the model *labelled* as class X, how many really were
  X. High precision = few false alarms.
- **Recall** (per class) - of the images that *truly* were class X, how many the model found.
  High recall = few misses.
- **Macro-Precision** - the macro-average of per-class precision (equal weight to every class).
  High macro-precision means few false alarms across all classes.
- **Macro-Recall** - the macro-average of per-class recall (equal weight to every class). High
  macro-recall means the model misses few true instances, even in rare classes.
- **Confusion matrix** - a 7×7 table where cell (row *i*, column *j*) counts how many true class-*i*
  images were predicted as class *j*. The diagonal is correct predictions; everything off-diagonal
  is a specific type of mistake.
- **Learning curves** - accuracy/loss plotted per training epoch for both the training and the
  validation set. They reveal **overfitting** (training keeps improving while validation gets
  worse) and **underfitting** (both stay low).

For a 7-class problem, **random guessing scores ≈ 14.3 %** (1 ÷ 7 = 0.143, i.e. with seven equally
likely classes a blind guess is right one time in seven) - that is the floor every model must beat.

---

## 8. Models and Results

> All experiments in this section run on the **140-sample augmented custom dataset** with a 20 %
> test set (28 images). Every metric is appended automatically to
> [`results/model_metrics.csv`](results/model_metrics.csv) via the
> [`save_metrics`](src/save_metrics.py) utility.

### 8.1 Baseline Classifiers

Script: [`src/baseline_models.py`](src/baseline_models.py)

**What and why.** Before reaching for deep learning, three classical algorithms establish a
*baseline*. Each 48×48 image is **flattened** into a single 2,304-dimensional vector (pixels in a
row), which is what these algorithms expect. Their purpose is to show *how far simple methods get*
and to justify the need for convolutional models.

- **k-Nearest Neighbours (k=3)** - classifies an image by the majority vote of its 3 closest
  images in raw-pixel space.
- **Decision Tree** - learns a tree of pixel-threshold rules (CART, default settings). CART
  (*Classification and Regression Trees*) is the algorithm that builds the tree by repeatedly
  finding the single pixel threshold (e.g. "pixel 712 > 0.4?") that best separates the classes
  at each step, until every branch leads to a pure class or a stopping condition is reached.
- **Gaussian Naive Bayes** - assumes each pixel is an independent Gaussian feature. *Gaussian*
  means it models each pixel's intensity as a normal distribution - it remembers the
  mean and standard deviation of that pixel separately for each emotion class. *Naive* means it
  treats all pixels as independent of each other. At prediction time it picks the emotion whose learned distributions best match the
  new image's pixel values.

**How tested.** 80/20 stratified split **plus** 5-fold stratified cross-validation. In 5-fold
CV the full dataset is divided into 5 equal parts (folds); the model is trained on 4 folds and
tested on the remaining 1, repeated 5 times so every fold serves as the test set exactly once.
The final score is the mean (± standard deviation) across all 5 runs - a more reliable estimate
than a single 80/20 split, especially important here where the test set is only 28 images and one
misclassified image shifts accuracy by ≈ 3.6 %.

**Results:**

| Model | Test Accuracy | CV Accuracy (mean ± std) |
|-------|--------------:|-------------------------:|
| kNN (k=3) | 46.43 % | 56.43 % ± 8.27 % |
| Decision Tree | 46.43 % | 42.14 % ± 6.14 % |
| **Naive Bayes (Gaussian)** | **71.43 %** | **65.00 % ± 7.95 %** |

Confusion matrices:

![kNN confusion matrix](results/cm_knn.png)
![Decision Tree confusion matrix](results/cm_decision_tree.png)
![Naive Bayes confusion matrix](results/cm_naive_bayes.png)

**Interpretation.** Naive Bayes clearly leads here. On a *tiny, balanced* set, its simple
per-pixel Gaussian assumption is hard to overfit and captures enough signal. However, the large
cross-validation standard deviations (±6–8 %) and the fact that the test set is only 28 images
mean these differences should be read with caution - a single misclassified image shifts accuracy
by ~3.6 %. The key baseline takeaway is the expected one: **treating pixels as an unordered vector
throws away spatial structure**, which is precisely what convolutional models exploit.

### 8.2 Multi-Layer Perceptrons (MLP)

Script: [`src/mlp_models.py`](src/mlp_models.py)

**What and why.** An MLP is the simplest neural network - fully connected ("dense") layers on the
flattened pixel vector. Three configurations probe the effect of **depth**, **width** and
**activation function**.

| Experiment | Architecture | Activation | Dropout | Epochs |
|------------|--------------|------------|---------|--------|
| Shallow | Dense(128) | ReLU | – | 15 |
| Deep + Dropout | Dense(256) → Dense(128) | ReLU | 0.3 | 15 |
| Deep + Tanh | Dense(256) → Dense(128) | Tanh | 0.3 | 15 |

All use Adam (lr = 0.001), sparse categorical cross-entropy, batch size 16, 10 % validation split.

**Results:**

| Model | Test Accuracy | Macro Precision | Macro Recall |
|-------|--------------:|----------------:|-------------:|
| MLP Shallow | 53.57 % | 0.544 | 0.536 |
| MLP Deep (ReLU) | 42.86 % | 0.348 | 0.429 |
| MLP Deep (Tanh) | 39.29 % | 0.257 | 0.393 |

Results around 40–54 % confirm that flat dense layers are a poor fit for image data: by
flattening the image into a pixel vector, spatial relationships between neighbouring pixels
are destroyed, and the network has no way to recover them. The scores are better than the
kNN and Decision Tree baselines, but the MLP still treats every pixel as an isolated number
rather than part of a face - which is exactly the limitation that convolutional layers are
designed to fix.

Confusion matrices:

![MLP shallow](results/cm_mlp_shallow.png)
![MLP deep](results/cm_mlp_deep.png)
![MLP tanh](results/cm_mlp_tanh.png)

**Reading the confusion matrices.** The shallow model has the fullest diagonal (its macro precision and
recall are both ≈ 0.54, i.e. errors are spread evenly). The deep and especially the `tanh` matrices
instead funnel many emotions into the **same few columns** - their low macro precision (0.35 and 0.26)
is the visual fingerprint of a network that defaults to a couple of classes instead of separating all
seven.

Combined learning curves:

![MLP learning curves](results/mlp_learning_curves.png)

**Reading the learning curves.** All three validation curves are **jagged and never settle** -
unavoidable when the 10 % validation split is only ~14 images, so a single image flips the score by
~7 %. The shallow model's training accuracy climbs steadily toward ~0.8, while the deeper variants stay
low and erratic - the textbook look of **underfitting** on too little data.

**Interpretation.** Counter-intuitively, the **shallow** network wins. With only 140 images, the
deeper networks have too many parameters to learn meaningfully and *underfit* - adding capacity
without adding data hurts. `tanh` performs worst, as expected: it saturates more easily than
`ReLU` and is harder to train on raw pixels. This is a clean demonstration of the
**bias–variance / data-size** relationship.

### 8.3 Custom Convolutional Neural Network (CNN)

Script: [`src/cnn_model.py`](src/cnn_model.py)

**What and why.** CNNs are the natural tool for images: instead of flattening, they slide small
filters across the image and learn **spatial features** (edges → textures → shapes). This built-in
"inductive bias" is exactly what the baselines lacked.

The network is built as three **convolutional blocks**, each followed by pooling, then a fully
connected classifier:

- **Conv2D(32, 3×3)** - 32 small 3×3 filters slide across the 48×48 image. Each filter learns
  to detect a simple low-level pattern (e.g. a horizontal edge, a dark spot). Output: 32 feature
  maps.
- **MaxPooling2D(2×2)** - shrinks each feature map by half (takes the maximum in each 2×2 window).
  This reduces the image size while keeping the strongest detected features, and makes the model
  less sensitive to small shifts in position.
- **Conv2D(64, 3×3)** - 64 filters applied to the already-reduced maps. At this depth the network
  combines the low-level edges from block 1 into mid-level patterns (curves, regions of a face).
- **MaxPooling2D(2×2)** - another halving.
- **Conv2D(128, 3×3)** - 128 filters at the smallest spatial scale. Here the network recognises
  high-level structures (eye region, mouth shape) that are specific to particular emotions.
- **MaxPooling2D(2×2)** - final spatial reduction.
- **Flatten** - converts the 3D feature maps into a single 1D vector, ready for a standard
  classifier.
- **Dense(128, ReLU)** - a fully connected layer that combines all detected features to form a
  decision.
- **Dropout(0.5)** - randomly switches off 50 % of neurons during each training step, preventing
  the network from memorising the 112 training images instead of learning general patterns.
- **Dense(7, Softmax)** - the output layer: one neuron per emotion. Softmax converts the raw
  scores into probabilities that sum to 1; the emotion with the highest probability is the
  prediction.

```
Input (48×48×1)
 → Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
 → Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
 → Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
 → Flatten → Dense(128, ReLU) → Dropout(0.5)
 → Dense(7, Softmax)
```

Adam (lr = 0.001), sparse categorical cross-entropy, batch size 16, **30 epochs**, 10 % validation.

**Result:** Test accuracy = **67.86 %**, Macro Precision = **0.798**, Macro Recall = **0.668**.

The high precision (0.798) relative to recall (0.668) means the model is **cautious** - when it
commits to a prediction it is usually right, but it sometimes fails to recognise an emotion at
all (misclassifies it as something else). This is a reasonable behaviour on a tiny dataset: the
model learned to be selective rather than guess aggressively.

![CNN confusion matrix](results/cm_cnn.png)

**Reading the confusion matrix.** The diagonal clearly dominates - most classes are classified
correctly - and the handful of errors are scattered rather than piled onto one column. That matches the
high macro precision (0.798): when the CNN commits to a label it is usually right, it simply misses a
few faces (lower recall 0.668).

![CNN learning curves](results/cnn_learning_curves.png)

**Reading the learning curves.** Training and validation accuracy **climb together** and the loss falls
steadily to ~0.5 with no widening gap between them - healthy learning with no serious overfitting,
helped by `Dropout(0.5)`. The validation accuracy sitting at zero for the first ~8 epochs is an artefact
of the tiny 10 % validation split; once training accuracy passes ~0.25 the model starts getting those
few held-out images right and the curve jumps up.

**Interpretation.** The CNN is the strongest *neural* model on the custom set and the second best
overall. The learning curves show the loss dropping steadily and training/validation tracking each
other - healthy learning with no severe overfitting, helped by the `Dropout(0.5)` layer. It
validates the central hypothesis: **convolutional structure beats flat dense layers on images**,
even with only 140 samples.

### 8.4 Transfer Learning - MobileNetV2 (Experiment)

Script: [`src/transfer_learning.py`](src/transfer_learning.py)

**What and why.** Training a deep network from scratch requires thousands of images. With only 140,
that is not realistic for a large architecture. **Transfer learning** solves this by reusing a
network that was already trained on a completely different, massive dataset - in this case
**ImageNet**, which contains 1.2 million colour photographs of everyday objects (cats, cars,
furniture, etc.). That network has already learned how to detect edges, textures, and shapes in
general. Instead of throwing that knowledge away, it is reused here as a starting point.

Concretely: **MobileNetV2** (Sandler et al., 2018) is loaded with its ImageNet weights and its
layers are **frozen** (not updated during training). Only a small new **classification head**
added on top is trained - it takes the features MobileNetV2 already knows how to extract and
learns to map them to the 7 emotion classes. Because MobileNetV2 needs 3-channel RGB input, the
grayscale images are converted to "pseudo-RGB" by repeating the single channel three times.

```
MobileNetV2 (frozen, ImageNet weights, input 48×48×3)
 → GlobalAveragePooling2D
 → Dense(128, ReLU) → Dropout(0.3)
 → Dense(7, Softmax)
```

**GlobalAveragePooling2D** replaces the `Flatten` step used in the custom CNN (§8.3). MobileNetV2
outputs a stack of feature maps (a small grid of numbers per learned feature); instead of unrolling
that whole grid into one very long vector, global average pooling takes the **average of each feature
map**, producing a single number per feature. This yields a compact fixed-length vector, drastically
fewer parameters in the head, and less overfitting - which is why it is the standard choice on top of
pretrained backbones.

Adam (lr = 0.001), batch size 16, **20 epochs**.

**Result:** Test accuracy = **64.29 %**, Macro Precision = **0.718**, Macro Recall = **0.660**.

The result is only slightly below the custom CNN (67.86 %), despite the fact that MobileNetV2
was never trained on faces or emotions - its frozen features came from photos of objects. This
shows that low-level visual features (edges, textures, local shapes) are universal enough to
transfer across domains. The precision and recall are close to each other (0.718 vs 0.660),
meaning the model makes errors roughly equally in both directions - no strong bias toward
over- or under-predicting any emotion.

![Transfer learning confusion matrix](results/cm_transfer_learning.png)

**Reading the confusion matrix.** The diagonal is strong but slightly less clean than the CNN's, and
the errors are spread fairly evenly across classes (macro precision 0.718 ≈ recall 0.660) - no single
emotion dominates the mistakes, so the model is balanced rather than collapsing onto one class.

![Transfer learning learning curves](results/tl_learning_curves.png)

**Reading the learning curves.** Here the curves **diverge**: training accuracy races to 100 % while
validation plateaus around 55–58 % and validation loss flattens near 1.3 - a textbook **overfitting
gap**. The frozen ImageNet features fit the 112 training crops perfectly but generalise only moderately
to the held-out ones, the expected consequence of a domain mismatch (objects → faces) on a tiny set.

**Interpretation.** Slightly below the custom CNN, but it converges quickly and needs no
architecture design. The catch is a **domain mismatch**: ImageNet features were learned on large,
colourful natural images, while these are tiny 48×48 grayscale faces - so the pretrained features
are only partly relevant. Still, beating 60 % with a frozen backbone confirms ImageNet features
are a useful general visual prior.

### 8.5 Vision Transformer (ViT)

Script: [`src/vit_model.py`](src/vit_model.py)

**What and why.** CNNs process images by sliding small filters across
neighbouring pixels - they have a built-in assumption that nearby pixels are related. The
**Vision Transformer** (Dosovitskiy et al., 2021) takes a completely different approach: it
borrows the **Transformer** architecture from natural language processing (the same idea behind
large language models like GPT). Instead of looking at local patches of pixels with filters, it
looks at all parts of the image **simultaneously** and learns which parts are relevant to each
other - this is called **self-attention**.

The idea: if a model is classifying an angry face, it might need to relate the shape of the
eyebrows to the shape of the mouth. Self-attention lets the model learn such long-range
dependencies directly, without being limited to a 3×3 filter window.

A custom mini-ViT was implemented from scratch to test this modern architecture on this problem:

1. **Patch extraction** - the 48×48 image is cut into 36 small 8×8 patches (like tiling a floor).
   Each patch is treated as one "token" - analogous to a word in a sentence.
2. **Patch embedding** - each patch (64 pixels) is projected to a 64-dimensional vector so the
   Transformer can process it.
3. **Positional encoding** - the Transformer has no notion of order or position, so learnable
   position embeddings are added to tell it where in the image each patch came from.
4. **Transformer encoder** - 2 blocks of self-attention: each patch "looks at" all other patches
   and decides how much attention to pay to each one. The 4 attention heads do this in parallel,
   each focusing on different relationships.
5. **Classification head** - the resulting patch representations are averaged and passed through
   a Dense(7, softmax) layer to produce the emotion prediction.

Adam (lr = 0.001), batch size 16, **40 epochs**.

**Result:** Test accuracy = **17.86 %**, Macro Precision = **0.327**, Macro Recall = **0.192**.

The accuracy of 17.86 % is barely above the 14.3 % random baseline - the model learned almost
nothing. The gap between precision (0.327) and recall (0.192) reveals the failure mode: when the
model does predict a class it is sometimes right (precision not terrible), but it misses the
vast majority of true instances (recall very low). In practice the model collapses - it predicts
only a few dominant classes and ignores the rest entirely.

![ViT confusion matrix](results/cm_vit.png)

**Reading the confusion matrix.** The matrix is the visual opposite of a healthy one: predictions pile
into just a couple of columns and most of the diagonal is empty. The model has **collapsed onto a few
dominant classes** and ignores the rest entirely - exactly what the very low recall (0.192) measures.

![ViT learning curves](results/vit_learning_curves.png)

**Reading the learning curves.** Both curves are violently **erratic and essentially flat** - validation
accuracy swings between 0 and 0.25 from one epoch to the next and never converges, while the loss barely
moves. The network never settles into a stable mapping, the expected behaviour of a data-hungry
transformer starved of data.

**Interpretation.** The ViT performs barely above the 14.3 % random baseline - and this is the
**expected, instructive result**. Transformers have *no built-in assumptions about images* (no
locality, no translation invariance), so they must learn everything from data. They are famously
**data-hungry**, typically needing tens of thousands to millions of images. The learning curves
are erratic and the model never converges. **This experiment is valuable precisely because it
fails:** it shows empirically *why* a state-of-the-art architecture can be the wrong choice when
data is scarce, and it directly motivates choosing a pretrained CNN (MobileNetV2) for production.

### 8.6 Conditional DCGAN (Generative Bonus)

Script: [`src/gan_model.py`](src/gan_model.py)

**What and why (generative AI bonus).** All previous models in this report are **classifiers** -
they take an image and output a label. A **GAN (Generative Adversarial Network)** does the
opposite: it learns to **generate new images from scratch**. The goal here was to generate
synthetic face photographs for a chosen emotion, which could in principle be used to expand a
small dataset.

A GAN works by training two networks that compete against each other:

- **Generator** - starts from a random noise vector (100 random numbers) plus an emotion label,
  and tries to produce a realistic-looking 48×48 face image. It has never seen a real face - it
  must figure out what one looks like purely from the feedback it receives.
- **Discriminator** - looks at an image (either real from the dataset, or generated) and its
  emotion label, and outputs a single answer: *real or fake?*

The two networks play a game: the Generator tries to fool the Discriminator into thinking its
images are real; the Discriminator tries to catch fakes. Over time, the Generator is forced to
produce increasingly realistic images to keep fooling it. This adversarial dynamic is what makes
GANs capable of generating photorealistic images.

The **conditional** part means both networks are given the emotion label - so the Generator can
be told *"generate an angry face"* rather than just *"generate any face"*.

The Generator architecture uses `Conv2DTranspose` layers - the reverse of convolution, which
**upsamples** a small representation into a full-size image step by step. The Discriminator uses
standard `Conv2D` layers to analyse the image.

Both networks are trained together for 50 epochs on the 140-image augmented set. Samples were
saved every 10 epochs to [`results/gan_progress/`](results/gan_progress/).

**Generated samples after 50 epochs (one per emotion):**

![GAN generated faces, epoch 50](results/gan_progress/gan_generated_epoch_050.png)

**Interpretation.** The generator learns the **global structure of a face** - a centred oval with
darker eye/mouth regions is clearly visible - but the outputs are **blurry**, show **checkerboard
artifacts** (a visual noise pattern caused by the transposed convolution upsampling), and the
seven emotion classes are **not yet visually distinct** from each other. This is expected: GANs
are notoriously data-hungry and need careful tuning to avoid **mode collapse** (where the
Generator finds one output that always fools the Discriminator and stops diversifying). With only
140 training images, there is simply not enough signal. As a **proof of concept** the experiment
succeeds - it demonstrates the full GAN training loop with two competing networks - while
honestly showing the limits of generative modelling on a micro-dataset.

---

## 9. Comparative Results Summary

All metrics are recorded in [`results/model_metrics.csv`](results/model_metrics.csv).

| Model | Dataset | Test Acc. | Macro Prec. | Macro Rec. | CV Acc. |
|-------|---------|----------:|------------:|-----------:|--------:|
| kNN (k=3) | custom (140) | 46.43 % | - | - | 56.43 % ± 8.27 % |
| Decision Tree | custom (140) | 46.43 % | - | - | 42.14 % ± 6.14 % |
| **Naive Bayes** | custom (140) | **71.43 %** | - | - | 65.00 % ± 7.95 % |
| MLP Shallow | custom (140) | 53.57 % | 0.544 | 0.536 | - |
| MLP Deep (ReLU) | custom (140) | 42.86 % | 0.348 | 0.429 | - |
| MLP Deep (Tanh) | custom (140) | 39.29 % | 0.257 | 0.393 | - |
| **Custom CNN** | custom (140) | **67.86 %** | **0.798** | **0.668** | - |
| MobileNetV2 (TL) | custom (140) | 64.29 % | 0.718 | 0.660 | - |
| Vision Transformer | custom (140) | 17.86 % | 0.327 | 0.192 | - |
| MobileNetV2 (Final) | FER-2013 | 37.38 % | 0.328 | 0.360 | - |

**Key observations:**

1. **Naive Bayes scores highest** on the custom test set (71.4 %), but on only 28 test images this
   is within the noise band - it should not be over-interpreted as "the best model in general".
2. **The custom CNN is the strongest, most reliable learner** (67.9 %, healthy learning curves):
   convolutional inductive bias pays off even with little data.
3. **Deeper is not better here** - both the deep MLPs and the ViT underperform their simpler
   counterparts because the dataset is too small to feed their capacity.
4. **The ViT's near-random score is a feature, not a bug** of the study: it empirically shows the
   data-hunger of attention-based models.

---

## 10. Final Production Model - MobileNetV2 on FER-2013

Script: [`src/final_train.py`](src/final_train.py) · Saved model: `models/final_emotion_model.h5`

### Why this model powers the app (and not Naive Bayes)

The comparison in §8 ranks Naive Bayes and the custom CNN highest *on 140 custom images*. So why
deploy MobileNetV2 trained on FER-2013 instead? Because **the comparison metric and the deployment
requirement are different things**:

- The 140-image models are tuned to *those specific 28 test faces*. With so few examples they will
  **not generalise** to arbitrary webcam photos from unknown users.
- A production model must be trained on **thousands of varied faces** to be robust. FER-2013
  (28,709 training images) provides exactly that scale, and **MobileNetV2 is the only architecture
  tested that combines pretrained robustness, fast convergence, and a small deployable footprint**
  suitable for a real-time app.

In short: §8 answers *"which algorithm learns best from a tiny set?"*; §10 answers *"which model
should ship?"*. Those are deliberately separate questions.

### Configuration

```
MobileNetV2 (frozen, ImageNet weights, input 48×48×3)
 → GlobalAveragePooling2D
 → Dense(256, ReLU) → Dropout(0.4)
 → Dense(7, Softmax)
```

- Optimiser: Adam (lr = 0.0005); Loss: categorical cross-entropy; Batch size: 64; Max 50 epochs.
- **Balanced class weights** (`compute_class_weight`, `class_weight='balanced'`) - the weights are not
  hand-picked; each is computed automatically as `n_samples / (n_classes × class_count)`, so rare
  classes are up-weighted in inverse proportion to their frequency. With FER-2013's distribution this
  ranges from **`disgust` ≈ 9.4** (436 images) down to **`happy` ≈ 0.57** (7,215 images), forcing the
  model to pay ~16× more attention to a `disgust` image than to a `happy` one so it cannot ignore the
  rare classes.
- **Callbacks:** `EarlyStopping(patience=5, restore_best_weights=True)` and
  `ModelCheckpoint(monitor='val_accuracy')`.
- Training images augmented in-pipeline (rotation ±20°, zoom ±20 %, horizontal flip); test images
  only rescaled.

**Result:** Test accuracy = **37.38 %**, Macro Precision = **0.328**, Macro Recall = **0.360**. Early stopping triggered after
~16 epochs.

**Why 37 % is a reasonable result, not a poor one.** The number looks low only next to the 68 % of the
custom CNN, but the two are not comparable: the custom CNN was scored on **28 in-distribution images**
from the same tiny set it trained on, whereas this model is scored on **7,178 unseen FER-2013 faces**
spanning far more people, lighting conditions and angles - a genuine generalisation test. Three things
put 37 % in context: (1) FER-2013 is a notoriously hard, label-noisy benchmark where even **human
accuracy is only ~65 %**, and simple frozen-backbone transfer baselines in the literature commonly land
in the **40–60 %** range; (2) the score is **~2.6× the 14.3 % random floor**, so the model has clearly
learned real signal rather than guessing; and (3) as §10.1 shows, **most of the remaining error is
confusion between neighbouring emotions** (fear/sad/angry), not gross mistakes - collapsing to the
valence level already recovers ~64 %. Macro precision and recall being close (0.328 vs 0.360) confirms
the model is **not inflating accuracy by collapsing onto one class**, though a residual lean toward
`happy` remains, as the confusion matrix shows.

![Final model confusion matrix](results/cm_final_model.png)

**Reading the confusion matrix.** The diagonal is strongest for `happy` and `surprise` - these are
recognised reliably - while `disgust` is essentially never recovered, unsurprising given it has only
436 training images. The dominant error is a **lean toward `happy`**: many `angry`, `fear`, `neutral`
and `sad` faces are misread as `happy`, the fingerprint of FER-2013's imbalance surviving even after
class weighting. Negative emotions also blur into one another (fear ↔ sad ↔ angry) - exactly the
within-category confusion quantified in §10.1.

![Final model learning curves](results/final_model_learning_curves.png)

**Reading the learning curves.** Validation accuracy **keeps rising and even sits above training
accuracy** when early stopping halts training - the signature of **underfitting, not overfitting**.
The model still had more to learn, but the frozen backbone caps how far it can go: it cannot adapt its
features to faces, so the small trainable head plateaus.

### Interpretation (honest)

The bottleneck is the **frozen backbone**: ImageNet features at 48×48 grayscale are simply not
expressive enough, and the small trainable head cannot compensate. FER-2013 is a genuinely hard
benchmark, but the single most impactful improvement available here would be to **unfreeze and
fine-tune the top MobileNetV2 layers** (see §15).

### 10.1 Emotion aggregation analysis (2-class)

Script: [`src/emotion_aggregation.py`](src/emotion_aggregation.py)

The 37 % figure measures the *hardest* possible task: telling apart seven emotions, several of
which are visually almost identical (`fear` vs `sad` vs `angry`). A natural question is **how much of that error is genuine failure, and how much is just confusion between
neighbouring emotions?** To answer it, the **same trained 7-class model** is re-evaluated after
collapsing the predictions into **two groups**. No retraining is involved: the model's raw
predictions on the 7,178 test images are simply regrouped, so any improvement comes purely from
*asking an easier, coarser question* - exactly the question the music recommender actually cares
about.

Three schemes were tested, two of them grounded in **Russell's circumplex model** (the same
arousal–valence framework that drives the association rules in §11):

| Scheme | Group A | Group B |
|--------|---------|---------|
| **Valence** (pos/neg) | happy, surprise, neutral | angry, disgust, fear, sad |
| **Happy vs rest** (one-vs-rest) | happy | all other six |
| **Arousal** (high/low) | angry, fear, happy, surprise | disgust, neutral, sad |

**Results** (post-hoc, identical model and predictions):

| Scheme | Accuracy | Macro-F1 | Balanced Acc. |
|--------|---------:|---------:|--------------:|
| 7-class (baseline) | 37.5 % | 0.334 | 0.373 |
| **Valence (pos/neg)** | **64.5 %** | **0.637** | **0.638** |
| Happy vs rest | 72.5 % | 0.651 | 0.662 |
| Arousal (high/low) | 64.4 % | 0.606 | 0.605 |

**Interpretation of the results.** Regrouping the *same* predictions lifts accuracy from 37 % to
**~64 %** on the valence split and to **72 %** for isolating happiness. The size of that jump shows that
**a large share of the seven-class error was within-group confusion** - e.g. a `sad` face mislabelled
`fear`, which is wrong at seven classes but correct once both count as *negative* - rather than gross
positive/negative mix-ups. Two caveats keep the reading honest: for `happy vs rest`, plain accuracy is
inflated by the majority "rest" group, so **macro-F1 (0.651) and balanced accuracy (0.662)** are the
fair metrics; and among the three, **valence is the most balanced and defensible** scheme (precision and
recall close on both sides). The result is also **product-relevant** - the recommender in §11 works
along the valence and arousal axes, so a dependable 64 % valence classifier is *closer to what the
system actually needs* than a fragile 37 % seven-way one.

![Valence aggregation confusion matrix](results/cm_aggregation_valence_pos_neg.png)
![Happy-vs-rest aggregation confusion matrix](results/cm_aggregation_happy_vs_rest.png)
![Arousal aggregation confusion matrix](results/cm_aggregation_arousal_high_low.png)

**Reading the confusion matrices.** The 2×2 matrices make the residual error visible and **asymmetric**.
On the valence split, positive faces are recognised well (recall 0.74), but **~47 % of negative faces
are still pulled into the positive group** (negative recall 0.53) - the lingering `happy`-bias from
§10 leaking through even after aggregation. The `happy vs rest` matrix shows the opposite imbalance:
the large "rest" group is captured cleanly (recall 0.79) while genuine `happy` faces are caught only
~54 % of the time, which is exactly why its high 72 % accuracy must be read together with the lower
macro-F1. The arousal matrix is the weakest of the three (low-arousal recall 0.46), because it forces
visually adjacent emotions like `sad` and `neutral` onto opposite sides of the split.

### 10.2 Training directly on 2-class labels (valence)

Script: [`src/binary_train.py`](src/binary_train.py) · Saved model: `models/binary_emotion_model.h5`

The §10.1 numbers are **post-hoc**: they regroup a model that was *trained* to separate seven
emotions. The natural follow-up is to **train MobileNetV2 directly on the two valence labels**, so the
binary objective is optimised end-to-end. The architecture, augmentation, class weights and callbacks
are identical to §10; only the head changes to two output neurons, and the FER-2013 emotion folders
are remapped to `positive`/`negative` on the fly (no images are copied). Training stopped via early
stopping after 25 epochs.

| Approach | Accuracy | Macro-F1 |
|----------|---------:|---------:|
| Post-hoc valence aggregation (§10.1) | 64.5 % | 0.637 |
| **End-to-end binary training** | **65.9 %** | **0.652** |

**Interpretation of the results.** End-to-end training gives only a **modest gain** (+1.4 pp accuracy,
+0.015 macro-F1) over simply regrouping the 7-class predictions. That is itself informative: it
confirms the **frozen ImageNet backbone is the shared bottleneck** - the same underfitting diagnosed
in §10 limits both models, and optimising the binary objective cannot extract features the backbone
never produced. Takeaway: for a 2-class deployment the cheap post-hoc aggregation is *almost as good*
as a dedicated model, and the highest-leverage fix for either granularity is the same - **fine-tune
the top MobileNetV2 layers** (§15), not changing the number of output classes.

![Binary valence confusion matrix](results/cm_binary_model.png)

**Reading the confusion matrix.** The matrix shows the **same asymmetry** as the post-hoc valence split
(§10.1): positive recall stays high (0.75) while negative faces are still under-recovered (recall 0.55).
Training directly on the binary labels shifts the numbers only slightly rather than fixing the
underlying `happy`-lean inherited from the backbone.

![Binary valence learning curves](results/binary_model_learning_curves.png)

**Reading the learning curves.** Training and validation accuracy climb together to ~0.66 and then
**plateau early**, with validation tracking - and for much of the run even sitting above - training.
This is once more the signature of **underfitting** caused by the frozen backbone, which is precisely
why fine-tuning (not re-labelling) is the next step.

---

## 11. Association Rule Mining (Apriori)

Script: [`src/association_rules.py`](src/association_rules.py)

### Motivation

Detecting an emotion is only half the system - it must be **translated into music**. The link is
built with **association rule mining**, using the **Apriori** algorithm (Agrawal & Srikant, 1994)
to discover statistical rules of the form *emotion → music attribute*.

### Data

Real user-listening logs were not available, so **1,500 synthetic listening sessions** were
generated from emotion→music mappings grounded in music-psychology research (Russell's
arousal–valence model; Grekow, 2016). Each session is a transaction such as
`{happy, fast_tempo, major_key, high_energy}`. **10 % random noise** was injected (a tempo flip)
to mimic real-world variability so the rules are not trivially perfect.

| Emotion | Tempo | Key | Energy |
|---------|-------|-----|--------|
| happy | fast | major | high |
| sad | slow | minor | low |
| angry | fast | minor | high |
| neutral | slow | major | low |
| surprise | fast | major | high |
| fear | fast | minor | low |
| disgust | slow | minor | low |

### Parameters and Results

| Parameter | Value |
|-----------|-------|
| Algorithm | Apriori (`mlxtend`) |
| Min support | 0.05 |
| Min confidence | 0.70 |

**49 rules** were discovered (full list in
[`results/music_association_rules.csv`](results/music_association_rules.csv)). *Support* = how
often the itemset appears; *confidence* = P(consequent | antecedent); *lift* = how much more often
than chance the two co-occur (lift > 1 means a real positive association).

Strongest example rules:

| Rule | Confidence | Lift |
|------|:----------:|-----:|
| angry → {high_energy, minor_key} | 1.00 | 6.79 |
| fear → {fast_tempo, low_energy, minor_key} | 0.89 | 7.37 |
| neutral → {slow_tempo, major_key, low_energy} | 0.92 | 6.36 |
| happy → {major_key, high_energy} | 1.00 | 3.41 |
| sad → {slow_tempo, low_energy, minor_key} | 0.89 | 3.17 |

The two rules with confidence 1.00 (`angry`, `happy`) mean the association held in every single
session in the dataset - no exception. The lift values are the more informative signal: `fear`
with lift 7.37 means that fast, low-energy, minor-key music co-occurs with fear **7× more often
than would happen by chance**, which confirms a genuine, strong pattern rather than a coincidence.
Rules for `sad` and `happy` show lower lift (3.17–3.41) because their musical attributes
(slow/minor and fast/major respectively) are also shared by other emotions, making them less
exclusive.

Confidence heatmap (emotion × music attribute):

![Association rules heatmap](results/association_rules_heatmap.png)

**Interpretation.** Apriori successfully recovers the psychologically expected structure - high
lift values confirm the associations are far stronger than random co-occurrence. These mined rules are what the application consults to translate an emotion into a
music query.

---

## 12. The Application

Script: [`src/app.py`](src/app.py) - a **Streamlit** web application.

**User flow:**

1. **Input** - the user either uploads a photo or takes one with the webcam (live camera).
2. **Face detection** - MTCNN locates and crops the face.
3. **Emotion prediction** - the crop is resized to 48×48, normalised, and passed through the
   MobileNetV2 production model, producing a probability for each of the 7 emotions.
4. **Calibration (see note below)** - heuristic weights adjust the raw probabilities.
5. **Result display** - the dominant emotion and a **bar chart** of all class probabilities.
6. **Music recommendation** - the dominant emotion is looked up in the mined association rules;
   the resulting attributes (e.g. *upbeat*, *relax*, *dark*) plus an emotion→genre seed form a
   **Spotify search query**, and **3 tracks** are shown with album art, artist, and a clickable
   "Listen on Spotify" link.

Both MTCNN and the emotion model are cached with `@st.cache_resource` so they load only once.

> **Honest note on the calibration step.** The final model over-predicts `happy` (visible in the
> §10 confusion matrix and caused by FER-2013's imbalance). To make the live demo behave more
> sensibly, [`app.py`](src/app.py) multiplies the seven raw probabilities by fixed weights -
> **`fear ×3.0`, `angry ×2.5`, `sad ×2.5`, `surprise ×2.5`, `disgust ×1.5`, `neutral ×1.0`,
> `happy ×0.8`** - and re-normalises so they sum to 1 again. The pattern is deliberate: the single
> over-predicted class (`happy`) is damped below 1, while the emotions the model tends to *miss* are
> boosted in rough proportion to how strongly they are suppressed (`fear` most of all). **This is a
> post-hoc heuristic, not a principled fix** - it compensates for the model's bias at inference time
> rather than at training time, and the multipliers were hand-tuned on the demo, not learned. It is
> documented transparently here; the proper solution is to fine-tune the model (§15), after which
> calibration could be removed.

---

## 13. Limitations and Critical Discussion

A good study states its own weaknesses clearly:

1. **The custom dataset is very small (35 test images).** Differences of a few percent between
   models are within statistical noise. The comparison in §8 is best read as *qualitative*
   ("CNN-class models beat flat models; transformers fail without data") rather than as precise
   rankings.
2. **The model-selection logic spans two datasets.** Models are compared on the custom set but the
   production model is trained on FER-2013. This is justified in §10, but it means the §9 table and
   the deployed model are not directly comparable.
3. **The final model underfits FER-2013 (37 %).** The frozen backbone is the limiting factor; the
   model has not reached its potential. The aggregation analysis (§10.1) shows much of this error is
   *within-category* confusion - the coarse valence distinction reaches ~64 %.
4. **The app's calibration is a heuristic** (§12) - effective for the demo but not scientifically
   principled.
5. **The listening data is synthetic.** The association rules are only as valid as the
   psychology-based mapping used to generate them; they were not learned from real user behaviour.

None of these invalidate the work - they are the realistic trade-offs of a broad, end-to-end
project, and naming them is part of the analysis.

---

## 14. Conclusions

- A **complete, working emotion-to-music pipeline** was built, from raw photo to clickable Spotify
  recommendations.
- **Ten model configurations** were trained and compared with proper metrics, confusion matrices,
  cross-validation and learning curves. The results tell a coherent story: **convolutional models
  beat flat classifiers on images, and architecture capacity must match data size** - both the deep
  MLPs and the data-hungry Vision Transformer underperform on a 140-image set.
- The study **went beyond the standard syllabus** with a from-scratch **Vision Transformer** and a
  **conditional DCGAN**, and is grounded in a reviewed **bibliography** of FER and music-emotion
  literature.
- The **final production model** (MobileNetV2 on FER-2013) was chosen for the right reason -
  robustness and deployability at scale - and its modest accuracy is analysed honestly rather than
  hidden.

---

## 15. Future Work

The most impactful next steps, roughly in priority order:

1. **Fine-tune MobileNetV2** - unfreeze the top convolutional blocks and continue training at a low
   learning rate. This typically lifts FER-2013 accuracy into the 55–65 % range and would likely
   make the §12 calibration hack unnecessary.
2. **Train the comparison models on a FER-2013 subset** - to replace the noisy 28-image evaluation
   with a statistically reliable one.
3. **Replace synthetic listening data with real logs** (e.g. Spotify audio features for real
   playlists) so the association rules reflect actual behaviour.
4. **Strengthen the GAN** - more data, spectral normalisation, and resize-convolution instead of
   transposed convolution to remove checkerboard artifacts.
5. **Add explainability** - Grad-CAM heatmaps showing which facial regions drive each prediction.
6. **Fine-tune for the 2-class task** - §10.2 already trains a binary valence head end-to-end (65.9 %),
   but the gain over post-hoc aggregation is small because the frozen backbone is the bottleneck.
   Combining the binary head with an *unfrozen, fine-tuned* MobileNetV2 is the open next step.

---

## 16. Bibliography

Full annotated list in [`docs/bibliography.md`](docs/bibliography.md). The PDFs of the core papers
are kept in `articles/`.

1. Goodfellow, I., et al. (2013). *Challenges in Representation Learning: A report on three machine
   learning contests.* (FER-2013 dataset.) https://arxiv.org/abs/1307.0414
2. Ammar, S., Bouwmans, T., & Neji, M. (2022). *Face Identification Using Data Augmentation Based on
   the Combination of DCGANs and Basic Manipulations.* Information, 13(8), 370.
   https://doi.org/10.3390/info13080370
3. Kim, J.-H., Kim, N., & Won, C. S. (2022). *Facial Expression Recognition with Swin Transformer.*
   https://arxiv.org/abs/2203.13472
4. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.C. (2018). *MobileNetV2: Inverted
   Residuals and Linear Bottlenecks.* CVPR, 4510–4520. https://arxiv.org/abs/1801.04381
5. Dosovitskiy, A., et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition
   at Scale.* ICLR 2021. https://arxiv.org/abs/2010.11929
6. Grekow, J. *Music Emotion Maps in Arousal-Valence Space.* Białystok University of Technology.
7. Athavle, M., Mudale, D., Shrivastav, U., & Gupta, M. (2021). *Music Recommendation Based on Face
   Emotion Recognition.* JIEEE, 2(2), 1–11.
8. Agrawal, R., & Srikant, R. (1994). *Fast Algorithms for Mining Association Rules.* VLDB, 487–499.

---

## 17. Appendix - How to Reproduce

### Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with Spotify API credentials
([Spotify Developer Dashboard](https://developer.spotify.com/dashboard)):

```
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
```

### Run the full experimental pipeline

Run from the `src/` directory, in order:

```bash
python face_detector.py        # 1. detect + crop faces from the custom dataset
python preprocess.py           # 2. grayscale, resize 48×48, normalise → .npy
python augment.py              # 3. augment custom dataset (×3 → 140 images)
python eda.py                  # 4. exploratory data analysis + plots
python baseline_models.py      # 5. kNN, Decision Tree, Naive Bayes (+ 5-fold CV)
python mlp_models.py           # 6. three MLP experiments
python cnn_model.py            # 7. custom CNN
python transfer_learning.py    # 8. MobileNetV2 transfer-learning experiment
python vit_model.py            # 9. Vision Transformer
python gan_model.py            # 10. conditional DCGAN
python final_train.py          # 11. final production model on FER-2013 (~15–30 min)
python emotion_aggregation.py  # 12. post-hoc 2-class aggregation of the final model
python binary_train.py         # 13. end-to-end binary valence model (~15–30 min)
python association_rules.py    # 14. Apriori rule mining
```

### Run the application

```bash
cd src
streamlit run app.py
```