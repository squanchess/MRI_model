"""
Evaluation utilities for pretrained DINO models.

Provides:
  - Feature extraction from frozen backbone
  - kNN classifier (no training needed, fast sanity check)
  - Linear probe (train a single linear layer on frozen features)

These are standard SSL evaluation protocols — not specific to any modality.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Tuple


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract CLS token features from a frozen backbone.

    Args:
        backbone: ViT backbone (should output (B, embed_dim) when num_classes=0).
        data_loader: DataLoader yielding dicts with "image" key (and optional "label").
        device: Device to run inference on.

    Returns:
        (features, labels) tuple. Labels is None if not present in data.
    """
    backbone.eval()
    backbone.to(device)

    all_features = []
    all_labels = []
    has_labels = None

    for batch in data_loader:
        # Handle both dict and tuple/list batch formats
        if isinstance(batch, dict):
            images = batch["image"].to(device)
            labels = batch.get("label", None)
        elif isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
            labels = batch[1] if len(batch) > 1 else None
        else:
            images = batch.to(device)
            labels = None

        if has_labels is None:
            has_labels = labels is not None

        features = backbone(images)
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        all_features.append(features.cpu())

        if has_labels and labels is not None:
            all_labels.append(labels.cpu())

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0) if has_labels else None

    return features, labels


@torch.no_grad()
def knn_evaluate(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
    num_classes: Optional[int] = None,
) -> dict:
    """Weighted kNN classifier evaluation.

    Uses cosine similarity with temperature-scaled voting.

    Args:
        train_features: (N_train, D) reference features.
        train_labels: (N_train,) integer labels.
        test_features: (N_test, D) query features.
        test_labels: (N_test,) ground truth labels.
        k: Number of nearest neighbors.
        temperature: Softmax temperature for distance weighting.
        num_classes: Number of classes (auto-detected if None).

    Returns:
        Dict with "accuracy" and "num_samples".
    """
    if num_classes is None:
        num_classes = int(max(train_labels.max(), test_labels.max()) + 1)

    # L2 normalize
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)

    correct = 0
    total = test_features.shape[0]

    # Process in chunks to avoid OOM on large datasets
    chunk_size = 256
    for i in range(0, total, chunk_size):
        chunk = test_features[i : i + chunk_size]

        # Cosine similarity
        sim = chunk @ train_features.t()  # (chunk, N_train)

        # Top-k
        topk_sim, topk_idx = sim.topk(k, dim=1)  # (chunk, k)
        topk_labels = train_labels[topk_idx]  # (chunk, k)

        # Temperature-weighted voting
        weights = (topk_sim / temperature).exp()  # (chunk, k)
        votes = torch.zeros(chunk.shape[0], num_classes, device=chunk.device)
        votes.scatter_add_(1, topk_labels, weights)

        preds = votes.argmax(dim=1)
        correct += (preds == test_labels[i : i + chunk_size]).sum().item()

    accuracy = correct / total
    return {"accuracy": accuracy, "num_samples": total}


class LinearProbe(nn.Module):
    """Linear probe for evaluating frozen features.

    Trains a single linear layer on top of frozen backbone features.

    Usage:
        features, labels = extract_features(backbone, train_loader)
        probe = LinearProbe(embed_dim, num_classes)
        probe.fit(features, labels, epochs=100)
        acc = probe.evaluate(test_features, test_labels)
    """

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> list:
        """Train the linear probe.

        Args:
            features: (N, D) frozen features.
            labels: (N,) integer labels.
            epochs: Training epochs.
            lr: Learning rate.
            batch_size: Batch size.
            verbose: Print progress.

        Returns:
            List of per-epoch loss values.
        """
        device = next(self.parameters()).device
        features = features.to(device)
        labels = labels.to(device)

        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        self.train()
        losses = []
        n = features.shape[0]

        for epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            count = 0

            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                logits = self(features[idx])
                loss = criterion(logits, labels[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * idx.shape[0]
                count += idx.shape[0]

            scheduler.step()
            avg_loss = epoch_loss / count
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Linear probe epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")

        return losses

    @torch.no_grad()
    def evaluate(self, features: torch.Tensor, labels: torch.Tensor) -> dict:
        """Evaluate linear probe accuracy.

        Returns:
            Dict with "accuracy" and "num_samples".
        """
        device = next(self.parameters()).device
        self.eval()
        logits = self(features.to(device))
        preds = logits.argmax(dim=1)
        correct = (preds == labels.to(device)).sum().item()
        return {"accuracy": correct / labels.shape[0], "num_samples": labels.shape[0]}
