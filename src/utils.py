import os
import random
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

def set_seed(seed: int = 42):
    """
    固定亂數種子，提升可重現性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 讓 cudnn 盡量可重現
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(dir_path: str):
    """
    若資料夾不存在就建立
    """
    os.makedirs(dir_path, exist_ok=True)


def dice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    計算 binary segmentation 的 Dice score

    Args:
        preds:   [B, H, W]，0/1 prediction
        targets: [B, H, W]，0/1 ground truth
        smooth:  避免分母為 0

    Returns:
        每張圖的 dice，shape = [B]
    """
    preds = preds.float()
    targets = targets.float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def mean_dice_from_sigmoid_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    logits:  [B,1,H,W]
    targets: [B,1,H,W]
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, 1, H, W]
        targets: [B, 1, H, W], float, values in {0,1}
        """
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def init_weights_he(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def multiclass_logits_to_preds(logits: torch.Tensor) -> torch.Tensor:
    """
    將模型輸出的 logits [B, C, H, W]
    轉成預測類別圖 [B, H, W]

    適用於 CrossEntropyLoss 的 2-class segmentation
    """
    preds = torch.argmax(logits, dim=1)
    return preds


def batch_dice_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    直接從 logits 與 target 計算每張圖的 Dice

    Args:
        logits:  [B, 2, H, W]
        targets: [B, H, W]

    Returns:
        per-image dice, shape [B]
    """
    preds = multiclass_logits_to_preds(logits)
    return dice_score(preds, targets, smooth=smooth)


def mean_dice_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6
) -> float:
    """
    計算一個 batch 的平均 Dice
    """
    dice_per_image = batch_dice_from_logits(logits, targets, smooth=smooth)
    return dice_per_image.mean().item()


class AverageMeter:
    """
    用來累積平均 loss / dice
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def save_checkpoint(
    save_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    best_val_dice: Optional[float] = None,
    history: Optional[Dict[str, List[float]]] = None,
):
    """
    儲存 checkpoint
    """
    ensure_dir(os.path.dirname(save_path))

    checkpoint = {
        "model_state_dict": model.state_dict()
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if best_val_dice is not None:
        checkpoint["best_val_dice"] = best_val_dice

    if history is not None:
        checkpoint["history"] = history

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
):
    """
    載入 checkpoint

    Returns:
        model
        optimizer
        checkpoint(dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from: {checkpoint_path}")
    return model, optimizer, checkpoint


def init_history() -> Dict[str, List[float]]:
    """
    初始化訓練紀錄
    """
    return {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": []
    }


def update_history(
    history: Dict[str, List[float]],
    train_loss: float,
    train_dice: float,
    val_loss: float,
    val_dice: float
) -> Dict[str, List[float]]:
    """
    更新每個 epoch 的訓練紀錄
    """
    history["train_loss"].append(train_loss)
    history["train_dice"].append(train_dice)
    history["val_loss"].append(val_loss)
    history["val_dice"].append(val_dice)
    return history


def save_prediction_mask(
    pred_mask: torch.Tensor,
    save_path: str
):
    """
    將單張預測 mask 存成圖片

    Args:
        pred_mask: [H, W]，值為 0/1
        save_path: 輸出路徑，例如 xxx.png
    """
    ensure_dir(os.path.dirname(save_path))

    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()

    pred_mask = pred_mask.astype(np.uint8) * 255
    image = Image.fromarray(pred_mask)
    image.save(save_path)


def save_batch_predictions(
    preds: torch.Tensor,
    filenames: List[str],
    save_dir: str,
    suffix: str = ".png"
):
    """
    將一個 batch 的 prediction masks 存成圖片

    Args:
        preds: [B, H, W]，值為 0/1
        filenames: 對應原圖檔名，例如 ['Abyssinian_1.jpg', ...]
        save_dir: 輸出資料夾
        suffix: 輸出副檔名，預設 .png
    """
    ensure_dir(save_dir)

    preds = preds.detach().cpu()

    for pred, filename in zip(preds, filenames):
        stem = os.path.splitext(filename)[0]
        save_path = os.path.join(save_dir, stem + suffix)
        save_prediction_mask(pred, save_path)


def count_parameters(model: torch.nn.Module) -> int:
    """
    計算可訓練參數數量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """
    簡單印出模型名稱與參數量
    """
    n_params = count_parameters(model)
    print(f"{model_name}:")
    print(f"Trainable parameters: {n_params:,}")