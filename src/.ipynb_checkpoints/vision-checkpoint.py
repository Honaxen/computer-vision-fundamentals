"""
vision.py — Shared utilities for computer-vision-fundamentals

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ──────────────────────────────────────────────
# 01 · Image Basics
# ──────────────────────────────────────────────

def show_images(images, titles=None, cmap="gray", cols=4, figsize=None):
    """Display a list of numpy arrays or tensors in a grid.

    Args:
        images  : list of H×W or H×W×C arrays / tensors
        titles  : optional list of strings
        cmap    : colormap (ignored for RGB images)
        cols    : number of columns in the grid
        figsize : override figure size
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3):   # CHW → HWC
                img = img.transpose(1, 2, 0)
            if img.ndim == 3 and img.shape[2] == 1:
                img = img.squeeze(2)
            ax.imshow(img, cmap=cmap if img.ndim == 2 else None)
            if titles:
                ax.set_title(str(titles[i]), fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def to_grayscale(image):
    """Convert an H×W×3 uint8 RGB image to H×W float grayscale [0, 1]."""
    image = np.array(image, dtype=np.float32)
    if image.ndim == 2:
        return image / 255.0
    return (0.2989 * image[..., 0] +
            0.5870 * image[..., 1] +
            0.1140 * image[..., 2]) / 255.0


def pixel_stats(image):
    """Print basic pixel statistics for a numpy image array."""
    arr = np.array(image, dtype=np.float32)
    print(f"Shape  : {arr.shape}")
    print(f"Dtype  : {arr.dtype}")
    print(f"Min    : {arr.min():.4f}")
    print(f"Max    : {arr.max():.4f}")
    print(f"Mean   : {arr.mean():.4f}")
    print(f"Std    : {arr.std():.4f}")


# ──────────────────────────────────────────────
# 02 · Filters & Edges
# ──────────────────────────────────────────────

def apply_filter(image, kernel):
    """Apply a 2-D convolution kernel to a grayscale image (no padding).

    Args:
        image  : H×W float numpy array
        kernel : k×k float numpy array

    Returns:
        Filtered image as float numpy array.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = (padded[i:i+kh, j:j+kw] * kernel).sum()
    return out


# Common kernels ──────────────────────────────

KERNELS = {
    "sobel_x": np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32),

    "sobel_y": np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]], dtype=np.float32),

    "sharpen": np.array([[ 0, -1,  0],
                          [-1,  5, -1],
                          [ 0, -1,  0]], dtype=np.float32),

    "blur":    np.ones((3, 3), dtype=np.float32) / 9,

    "emboss":  np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]], dtype=np.float32),
}


def edge_magnitude(image):
    """Compute gradient magnitude using Sobel X + Y kernels."""
    gx = apply_filter(image, KERNELS["sobel_x"])
    gy = apply_filter(image, KERNELS["sobel_y"])
    return np.sqrt(gx**2 + gy**2)


def compare_filters(image, kernel_names=None):
    """Side-by-side comparison of multiple filter outputs.

    Args:
        image        : H×W float grayscale array
        kernel_names : subset of KERNELS keys; defaults to all
    """
    names = kernel_names or list(KERNELS.keys())
    results = [apply_filter(image, KERNELS[n]) for n in names]
    show_images(
        [image] + results,
        titles=["original"] + names,
        cmap="gray",
        cols=len(names) + 1
    )


# ──────────────────────────────────────────────
# 03 · CNN From Scratch
# ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    """Lightweight CNN for 28×28 grayscale classification (e.g. MNIST).

    Architecture:
        Conv1 (1→32) → Conv2 (32→64) → Pool
        Conv3 (64→128) → Pool
        FC (128*7*7 → 256) → FC (256 → num_classes)
    """

    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),           # 28 → 14
            nn.Dropout2d(dropout / 2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),           # 14 → 7
            nn.Dropout2d(dropout / 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ──────────────────────────────────────────────
# 04 · Image Classification — training helpers
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one full training epoch.

    Returns:
        (avg_loss, accuracy) as floats
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on a data loader (no gradient).

    Returns:
        (avg_loss, accuracy) as floats
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def fit(model, train_loader, val_loader, optimizer, criterion, device,
        num_epochs=10, scheduler=None, verbose=True):
    """Full training loop.

    Returns:
        history dict with keys: train_loss, train_acc, val_loss, val_acc
    """
    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if scheduler:
            scheduler.step()

        if verbose:
            print(f"Epoch {epoch+1:>3}/{num_epochs} "
                  f"| Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.2%} "
                  f"| Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.2%}")

    return history


# ──────────────────────────────────────────────
# Plotting helpers (used by 03, 04, 05)
# ──────────────────────────────────────────────

def plot_curves(history, title="Training Curves"):
    """Plot loss and accuracy curves from a history dict."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"],   label="Validation")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def show_predictions(model, loader, device, classes=None, n=16, cols=8):
    """Show a grid of images with true vs predicted labels.

    Args:
        model   : trained PyTorch model
        loader  : DataLoader (val or test)
        device  : torch.device
        classes : list of class name strings; falls back to integer labels
        n       : number of images to show
        cols    : columns in the grid
    """
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        preds = model(images).argmax(1)

    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    axes = np.array(axes).flatten()

    # ImageNet denorm constants (safe default; no-op if images aren't normalised this way)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, ax in enumerate(axes):
        if i < n:
            img = images[i].cpu()
            c = img.shape[0]
            if c == 3:
                img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            else:
                img = img.squeeze().numpy()

            true_lbl = classes[labels[i].item()] if classes else labels[i].item()
            pred_lbl = classes[preds[i].item()]  if classes else preds[i].item()
            color = "green" if preds[i] == labels[i] else "red"

            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(f"T:{true_lbl}\nP:{pred_lbl}", color=color, fontsize=8)
        ax.axis("off")

    plt.suptitle("Predictions — green: correct   red: wrong", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, loader, device, num_classes=10, class_names=None):
    """Compute and plot a confusion matrix for the full loader."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    labels = class_names or list(range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Validation Set", fontsize=13)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 05 · Transfer Learning helpers
# ──────────────────────────────────────────────

def build_transfer_model(num_classes, freeze_backbone=True):
    """Load pretrained ResNet-18 and replace the final layer.

    Args:
        num_classes      : number of output classes
        freeze_backbone  : if True, freeze all layers except fc

    Returns:
        model (nn.Module) — ready to move to device
    """
    from torchvision import models

    model = models.resnet18(weights="IMAGENET1K_V1")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def unfreeze_layer(model, layer_name):
    """Unfreeze a named layer for deeper fine-tuning.

    Example:
        unfreeze_layer(model, "layer4")

    After unfreezing, rebuild the optimizer:
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    """
    layer = getattr(model, layer_name, None)
    if layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in model.")
    for param in layer.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfroze '{layer_name}' — trainable params: {trainable:,}")


def param_summary(model):
    """Print total / trainable / frozen parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"Total      : {total:,}")
    print(f"Trainable  : {trainable:,}  ({trainable/total:.1%})")
    print(f"Frozen     : {frozen:,}")