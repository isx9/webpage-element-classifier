# webpage-element-classifier

Multi-label classification of UI element types from full webpage screenshot images, using frozen ResNet50 embeddings and classical ML classifiers.

**Course:** Data Mining & Machine Learning — Final Assignment  
**Dataset:** [Website Screenshots](https://public.roboflow.com/object-detection/website-screenshots) — 1,206 screenshots from the world's top websites (Roboflow, MIT License)

---

## Objective

Given a screenshot of a webpage, predict which UI element types are visually present:

| Label | Description |
|---|---|
| `button` | Navigation links, tabs, interactive controls |
| `heading` | Text enclosed in `<h1>` to `<h6>` tags |
| `link` | Inline textual `<a>` tags |
| `label` | Text labeling form fields |
| `text` | All other text content |
| `image` | `<img>`, `<svg>`, `<video>` tags and icons |
| `iframe` | Ads and third-party embedded content |
| `field` | Form input fields |

This is a **multi-label classification** problem: each screenshot can have multiple element types present simultaneously (average 5.3 labels per image).

## Non-trivial challenges

- **Severe class imbalance** — `label` appears in 5.4% of images, `text` in 97.7%
- **Multi-label output** — each image has on average 5.3 active labels out of 8
- **High-dimensional image input** — managed via frozen CNN transfer learning (no GPU required)

## Pipeline
screenshot → ResNet50 (frozen, ImageNet) → 2048-dim embedding → OneVsRest classifier → [button, heading, ..., iframe]

## Methods

- **Feature extraction** — frozen ResNet50 pretrained on ImageNet, 2048-dim output
- **EDA** — PCA, MDS, k-means, NMF on label matrix, Isolation Forest, hierarchical clustering
- **Models** — Logistic Regression, SVM, Random Forest, XGBoost wrapped in OneVsRest
- **Evaluation** — Hamming loss, micro/macro F1, per-class ROC-AUC, precision-recall curves
- **Interpretation** — saliency maps, per-class feature importance, failure case analysis

## Project structure

webpage-element-classifier/

├── data/

│   └── raw/               # original dataset (gitignored)

├── embeddings/            # extracted .npy feature matrices (gitignored)

├── figures/               # saved plots

├── notebook.ipynb         # main analysis notebook

├── requirements.txt

└── README.md

## Setup

```bash
git clone https://github.com/<your-username>/webpage-element-classifier.git
cd webpage-element-classifier
pip install -r requirements.txt
```

Download the dataset from [Roboflow](https://public.roboflow.com/object-detection/website-screenshots) and extract into `data/raw/`.
