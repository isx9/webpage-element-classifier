# Webpage UI Element Classifier

![Python](https://img.shields.io/badge/python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ResNet50-FF6F00?logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/status-academic%20project-lightgrey)

Multi-label classification of UI element types from full webpage screenshot images, using frozen ResNet50 embeddings and classical ML classifiers. This is useful when only a rendered image is available (e.g. vision-based browser agents, archived pages) and there's no DOM to parse directly.

**Course:** Data Mining & Machine Learning
**Dataset:** [Website Screenshots](https://public.roboflow.com/object-detection/website-screenshots) — 1,206 screenshots from the world's top websites (Roboflow, MIT License)

---

## Table of Contents

- [Objective](#objective)
- [Non-trivial Challenges](#non-trivial-challenges)
- [Pipeline](#pipeline)
- [Methods](#methods)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup](#setup)


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

This is a **multi-label classification** problem: each screenshot can have multiple element types present simultaneously (average ~5.3 labels per image).

## Non-trivial Challenges

- **Severe class imbalance** — `label` appears in ~5.4% of images, `text` in ~97.7%
- **Multi-label output** — each image has ~5.3 active labels out of 8, on average
- **Label correlation** — element types co-occur by design convention (e.g. `field` and `label` both appear on form-heavy pages) rather than independently
- **High-dimensional image input** — managed via frozen CNN transfer learning (no GPU required)


## Pipeline
screenshot → ResNet50 (frozen, ImageNet) → 2048-dim embedding → OneVsRest (LR / RF — labels treated independently) → Classifier Chain (LR — labels treated as correlated) → [button, heading, ..., iframe]

Two label-modeling strategies are compared on the same embeddings: OneVsRest, which trains each label independently, and a Classifier Chain, which lets each label's classifier condition on the others — directly testing whether the label co-occurrence found in the EDA is actually exploitable.

## Methods

- **Feature extraction** — frozen ResNet50 pretrained on ImageNet, 2048-dim output
- **EDA** — PCA, t-SNE, k-means (+ a fuzzy c-means check), NMF on the label matrix, outlier detection (Isolation Forest, LOF), hierarchical clustering
- **Models**
  - Logistic Regression and Random Forest, each wrapped in OneVsRest and tuned with GridSearchCV (5-fold cross-validation)
  - Classifier Chain using the same tuned Logistic Regression, averaged over 10 random label orderings, to isolate the effect of modeling label dependencies
- **Evaluation** — Hamming loss, micro/macro F1, per-class ROC-AUC, precision-recall curves
- **Interpretation** — LR coefficient sparsity, Random Forest feature importance, failure case analysis

## Results

Logistic Regression and Random Forest reach comparable overall performance (macro-F1 ≈ 0.70–0.73). Random Forest achieves higher overall accuracy (micro-F1 0.90) by performing very well on common classes (`text`, `image`, `button`), while Logistic Regression handles rare classes more evenly. Both models perform almost perfectly on distinctive, common elements and struggle on rare or visually ambiguous ones (`label`, `iframe`).

The **Classifier Chain** improves on every metric over the OneVsRest Logistic Regression baseline (Macro-F1 0.7333 → 0.7365, Micro-F1 0.8504 → 0.8547), confirming that modeling label dependencies helps. The
improvement is concentrated exactly where the EDA predicted: `field` (F1 0.67 → 0.69), one of the classes shown to co-occur with others. It does not help `label` or `iframe`, whose extreme rarity in the test set (6 and 26 instances) leaves too little signal for any method — chain or independent — to learn a reliable relationship. The main bottleneck for this dataset is data volume for the rarest classes, not model or strategy choice.


## Project structure

```
webpage-element-classifier/
├── data/
│ └── raw/ # original dataset (gitignored)
├── embeddings/ # extracted .npy feature matrices (gitignored)
├── figures/ # saved plots
├── webpage_ui_element_classification.ipynb # main analysis notebook
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/isx9/webpage-element-classifier.git
cd webpage-element-classifier
pip install -r requirements.txt
```

Download the dataset from [Roboflow](https://public.roboflow.com/object-detection/website-screenshots) and extract into `data/raw/`.
