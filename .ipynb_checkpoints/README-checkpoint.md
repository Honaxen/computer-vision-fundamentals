# Computer Vision Fundamentals

Building intuition for computer vision — from pixel manipulation to deep CNNs.
Every concept implemented from scratch before using a library.

---

## Why This Repo?

Computer vision is not just about calling a model.
Understanding how images are represented, filtered, and classified
makes you a better engineer when things go wrong.

---

## Structure

```
computer-vision-fundamentals/
├── notebooks/
│   ├── 01_image_basics.ipynb
│   ├── 02_filters_and_edges.ipynb
│   ├── 03_cnn_from_scratch.ipynb
│   ├── 04_image_classification.ipynb
│   └── 05_transfer_learning.ipynb
├── src/
│   └── vision.py
├── data/
│   ├── raw/
│   └── processed/
└── README.md
```

---

## Topics

| Notebook | Concept | Key Question |
|----------|---------|-------------|
| 01 | Image Basics | How are images represented as numbers? |
| 02 | Filters & Edges | How do filters detect features? |
| 03 | CNN From Scratch | How does a CNN learn to see? |
| 04 | Image Classification | How do we classify images with PyTorch? |
| 05 | Transfer Learning | How do we reuse a pretrained model? |

---

## Stack

Python · PyTorch · OpenCV · numpy · matplotlib

---

## What I Learned

Images are 3D arrays — (height, width, channels).
Every pixel is a number. Every filter is a learned transformation.

Gaussian blur removes noise by averaging neighbors.
Sobel filters detect edges by finding rapid pixel changes.
These hand-crafted filters became the templates that CNNs learn automatically.

A CNN trained from scratch on MNIST achieves 99.3% accuracy in 5 epochs.
The same accuracy with transfer learning requires only 5,130 trainable parameters — not 421,642.

Transfer learning is one of the most practical insights in modern deep learning:
pretrained features transfer across domains, even from color images to grayscale digits.

---

## Author

[Honaxen](https://github.com/Honaxen)