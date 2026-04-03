"""
预训练 DINO 模型的评估工具。

提供以下能力：
  - 从冻结 backbone 中提取特征
  - kNN 分类器评估（无需训练，适合快速 sanity check）
  - 线性探针评估（在冻结特征上训练单层线性分类器）

这些都是标准的自监督学习评估协议，并不依赖特定模态。
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """从冻结的 backbone 中提取 CLS token 特征。

    参数：
        backbone: ViT 主干网络；当 `num_classes=0` 时应输出 `(B, embed_dim)`。
        data_loader: 产出带 `"image"` 键（可选 `"label"`）的 DataLoader。
        device: 推理设备。

    返回：
        `(features, labels)` 元组；若数据中没有标签，则 `labels` 为 None。
    """
    backbone.eval()
    backbone.to(device)

    all_features = []
    all_labels = []
    has_labels = None

    for batch in data_loader:
        # 同时兼容 dict、tuple、list 这几种 batch 形式
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
    """加权 kNN 分类评估。

    使用余弦相似度，并通过 temperature 对投票权重进行缩放。

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

    # 做 L2 归一化
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)

    correct = 0
    total = test_features.shape[0]

    # 分块处理，避免大数据集导致显存/内存溢出
    chunk_size = 256
    for i in range(0, total, chunk_size):
        chunk = test_features[i: i + chunk_size]

        # 计算余弦相似度
        sim = chunk @ train_features.t()

        # 取 Top-k 近邻
        topk_sim, topk_idx = sim.topk(k, dim=1)
        topk_labels = train_labels[topk_idx]

        # 基于 temperature 的加权投票
        weights = (topk_sim / temperature).exp()
        votes = torch.zeros(chunk.shape[0], num_classes, device=chunk.device)
        votes.scatter_add_(1, topk_labels, weights)

        preds = votes.argmax(dim=1)
        correct += (preds == test_labels[i: i + chunk_size]).sum().item()

    accuracy = correct / total
    return {"accuracy": accuracy, "num_samples": total}


class LinearProbe(nn.Module):
    """用于评估冻结特征的线性探针。

    在冻结 backbone 特征之上训练一个线性层。

    用法：
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
        """训练线性探针。"""
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
                idx = perm[i: i + batch_size]
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
        """评估线性探针的准确率。"""
        device = next(self.parameters()).device
        self.eval()
        logits = self(features.to(device))
        preds = logits.argmax(dim=1)
        correct = (preds == labels.to(device)).sum().item()
        return {"accuracy": correct / labels.shape[0], "num_samples": labels.shape[0]}
