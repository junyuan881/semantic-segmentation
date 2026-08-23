import os
import csv
import argparse
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader

from oxford_pet import get_dataloader
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import (
    set_seed,
    load_checkpoint,
    print_model_summary,
)


def get_args():
    parser = argparse.ArgumentParser(description="Inference for binary semantic segmentation")

    parser.add_argument("--data_root", type=str, default="../dataset",
                        help="Oxford-IIIT Pet dataset root")
    parser.add_argument("--split_dir", type=str, default="../dataset/splits",
                        help="Directory containing train.txt / val.txt / test_xxx.txt")

    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "resnet34_unet"],
                        help="Model type")
    # parser.add_argument("--test_type", type=str, default="unet",
    #                     choices=["unet", "res_unet"],
    #                     help="Which test txt to use")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pth)")

    # parser.add_argument("--image_size", type=int, default=256,
    #                     help="Resize image to image_size x image_size")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    # parser.add_argument("--download", action="store_true",
    #                     help="Download dataset if needed")

    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to output submission csv")

    return parser.parse_args()


def build_model(model_name: str):
    if model_name == "unet":
        model = UNet(
            in_channels=3,
            n_classes=1,
            # base_channels=64,
            # bilinear=False
        )
    elif model_name == "resnet34_unet":
        model = ResNet34_UNet(
            in_channels=3,
            out_channels=1,
            use_cbam = True
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def logits_to_preds(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, 2, H, W]
    return: [B, H, W], 0/1
    """
    if logits.size()[1] == 2:
        preds = torch.argmax(logits, dim=1)
    elif logits.size()[1] == 1:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).squeeze(1)
    return preds


def rle_encode(mask: np.ndarray) -> str:
    """
    將 binary mask 轉成 RLE 字串
    要求：
    - foreground = 1
    - background = 0
    - column-major order (Fortran order)

    Args:
        mask: [H, W], np.uint8, values in {0,1}

    Returns:
        RLE string
    """
    mask = mask.astype(np.uint8)

    # Fortran order: 先按 column 再按 row
    pixels = mask.flatten(order="F")

    # 前後補 0，方便找 run 起點終點
    padded = np.concatenate([[0], pixels, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0] + 1

    starts = changes[::2]
    ends = changes[1::2]
    lengths = ends - starts

    if len(starts) == 0:
        return ""

    rle = " ".join(f"{s} {l}" for s, l in zip(starts, lengths))
    return rle
def rle_decode(rle: str, shape):
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)

    if rle.strip() == "":
        return mask.reshape((h, w), order="F")

    s = list(map(int, rle.split()))
    starts = s[0::2]
    lengths = s[1::2]

    for start, length in zip(starts, lengths):
        start -= 1   # RLE 是 1-based index
        mask[start:start + length] = 1

    return mask.reshape((h, w), order="F")

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()

    results = []

    for images,_, shapes, names in loader:
        images = images.to(device)
        orig_ws, orig_hs = shapes
        # print(images, masks, orig_ws, orig_hs, names)
        # print(orig_ws)

        logits = model(images)
        preds = logits_to_preds(logits)
        preds = preds.detach().cpu().numpy().astype(np.uint8)

        for pred_mask, filename, orig_w, orig_h in zip(preds, names, orig_ws, orig_hs):
            orig_w = int(orig_w)
            orig_h = int(orig_h)
            # print(pred_mask.size)
            pred_pil = Image.fromarray(pred_mask)
            pred_pil = pred_pil.resize((orig_w, orig_h), resample=Image.NEAREST)
            pred_mask_resized = np.array(pred_pil, dtype=np.uint8)

            image_id = os.path.splitext(filename)[0]
            encoded_mask = rle_encode(pred_mask_resized)

            results.append((image_id, encoded_mask))

    return results

def save_submission_csv(results, output_csv: str):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True) if os.path.dirname(output_csv) else None

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "encoded_mask"])
        writer.writerows(results)

    print(f"Submission saved to: {output_csv}")
    print(f"Total rows: {len(results)}")


def main():
    args = get_args()

    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switch to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.model=='unet':
        image_size = 388
    else:
        image_size = 224
    _, _, test_loader = get_dataloader(args.data_root+'/images',args.data_root+'/annotations/trimaps',args.split_dir,model = args.model,batch_size=args.batch_size,img_size=image_size)
    # _, _, test_loader = get_dataloader(args.data_root+'/images',args.data_root+'/annotations/trimaps',args.split_dir,model = args.model,batch_size=args.batch_size)


    model = build_model(args.model).to(device)
    print_model_summary(model, model_name=args.model)

    model, _, checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        optimizer=None,
        device=device
    )

    print(f"Checkpoint     : {args.checkpoint}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if "best_val_dice" in checkpoint:
        print(f"Checkpoint best_val_dice: {checkpoint['best_val_dice']:.4f}")
    print(f"Device         : {device}")
    print("=" * 60)

    results = run_inference(
        model=model,
        loader=test_loader,
        device=device
    )

    save_submission_csv(results, args.output_csv)


if __name__ == "__main__":
    main()

# python3 inference.py --data_root ../dataset/oxford-iiit-pet --split_dir ../dataset/splits --model unet --checkpoint ../saved_models/unet/best.pth --batch_size 8 --device cuda --output_csv ../submissions/unet_submission.csv
# python3 inference.py --data_root ../dataset/oxford-iiit-pet --split_dir ../dataset/splits --model resnet34_unet --checkpoint ../saved_models/res_unet/best.pth --batch_size 8 --device cuda --output_csv ../submissions/res_unet_submission.csv