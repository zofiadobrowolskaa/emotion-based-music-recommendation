# Facial Emotion Recognition for Music Recommendation

An emotion-aware music recommendation system. It detects a person's emotion from a facial
photograph using computer vision and deep learning, then recommends music whose audio
characteristics match that emotion through the Spotify API.

> **Full research report:** see **[REPORT.md](REPORT.md)** — datasets, preprocessing, every model,
> all results, confusion matrices, learning curves, and discussion.

---

## What it does

```
 Face photo → MTCNN face detection → 48×48 crop → emotion model (MobileNetV2)
            → Apriori rules (emotion → tempo/key/energy) → Spotify song recommendations
```

A face image is detected and cropped with **MTCNN**, classified into one of seven emotions
(*angry, disgust, fear, happy, neutral, sad, surprise*) by a **MobileNetV2** model, mapped to
musical attributes via **Apriori association rules**, and turned into live track recommendations
through the **Spotify API** — all inside a **Streamlit** web app.

## Highlights

- **10 model configurations** compared with accuracy, macro-F1, confusion matrices, learning
  curves and 5-fold cross-validation.
- Classical baselines (**kNN, Decision Tree, Naive Bayes**), three **MLPs**, a custom **CNN**,
  **MobileNetV2** transfer learning, and a from-scratch **Vision Transformer**.
- **Conditional DCGAN** for emotion-conditioned face generation (generative bonus).
- **Apriori** association-rule mining linking emotions to music attributes.
- Own balanced **custom dataset** (35 photos) alongside the **FER-2013** benchmark (35,887 images).

## Headline results

| Model | Dataset | Test Acc. | Macro F1 |
|-------|---------|----------:|---------:|
| Naive Bayes | custom (140) | 71.43 % | 0.697 |
| Custom CNN | custom (140) | 67.86 % | 0.675 |
| MobileNetV2 (transfer) | custom (140) | 64.29 % | 0.646 |
| MLP Shallow | custom (140) | 53.57 % | 0.524 |
| Vision Transformer | custom (140) | 17.86 % | 0.147 |
| **MobileNetV2 (production)** | **FER-2013** | **37.38 %** | **0.326** |
| MobileNetV2 — valence aggregation (pos/neg) | FER-2013 | 64.47 % | 0.637 |
| MobileNetV2 — binary valence (end-to-end) | FER-2013 | 65.88 % | 0.652 |

> The 37 % seven-class score rises to **~64 %** when the *same* model's predictions are aggregated
> into a 2-class valence split (positive vs negative) — most of the error is confusion *within* an
> emotional category, not across it. Training a model directly on the 2-class labels reaches **65.9 %**.
> See [§10.1](REPORT.md#101-emotion-aggregation-analysis-2-class)–[§10.2](REPORT.md#102-training-directly-on-2-class-labels-valence).

Full table and per-model analysis in [REPORT.md](REPORT.md). All metrics are aggregated in
[`results/model_metrics.csv`](results/model_metrics.csv).

## Project structure

```
emotion-based-music-recommendation/
├── data/                       # FER-2013, custom_dataset, processed arrays (gitignored)
├── models/final_emotion_model.h5
├── results/                    # confusion matrices, learning curves, heatmaps, CSV metrics
│   └── gan_progress/           # DCGAN samples per epoch
├── src/
│   ├── face_detector.py        # MTCNN face extraction (object detection)
│   ├── preprocess.py           # resize + grayscale + normalise → .npy
│   ├── augment.py              # data augmentation (rotation/zoom/flip/noise)
│   ├── eda.py                  # exploratory data analysis
│   ├── baseline_models.py      # kNN, Decision Tree, Naive Bayes + cross-validation
│   ├── mlp_models.py           # three MLP experiments
│   ├── cnn_model.py            # custom CNN
│   ├── transfer_learning.py    # MobileNetV2 transfer-learning experiment
│   ├── vit_model.py            # custom Vision Transformer
│   ├── gan_model.py            # conditional DCGAN
│   ├── final_train.py          # production training on FER-2013
│   ├── emotion_aggregation.py  # 2-class aggregation analysis (valence/arousal/one-vs-rest)
│   ├── binary_train.py         # end-to-end binary valence training (pos/neg)
│   ├── association_rules.py    # Apriori rule mining
│   ├── save_metrics.py         # shared metrics-logging utility
│   └── app.py                  # Streamlit web application
├── docs/bibliography.md
├── REPORT.md                   # full research report
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Spotify API credentials
([Spotify Developer Dashboard](https://developer.spotify.com/dashboard)):

```
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
```

## Usage

**Run the application:**

```bash
cd src
streamlit run app.py
```

**Reproduce the experiments** — run the scripts in `src/` in order (`face_detector.py` →
`preprocess.py` → `augment.py` → `eda.py` → the model scripts → `association_rules.py`). See the
[reproduction appendix in REPORT.md](REPORT.md#17-appendix--how-to-reproduce) for the full sequence.

## Bibliography

Eight references covering facial emotion recognition, MobileNetV2, Vision Transformers, DCGANs,
music-emotion mapping and the Apriori algorithm — see [`docs/bibliography.md`](docs/bibliography.md)
and the [bibliography section of REPORT.md](REPORT.md#16-bibliography).
