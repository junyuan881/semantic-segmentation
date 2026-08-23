import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import (
    set_seed,
    AverageMeter,
    mean_dice_from_logits,
    load_checkpoint,
    print_model_summary,
    mean_dice_from_sigmoid_logits,
    BCEDiceLoss
)
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")

    parser.add_argument("--data_root", type=str, default="../dataset",
                        help="Oxford-IIIT Pet dataset root")
    parser.add_argument("--split_dir", type=str, default="../dataset/splits",
                        help="Directory containing train.txt / val.txt / test_xxx.txt")
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "resnet34_unet"],
                        help="Model type")
    # parser.add_argument("--test_type", type=str, default="unet",
    #                     choices=["unet", "res_unet"],
    #                     help="Which test txt to use when building dataset")
    parser.add_argument("--eval_split", type=str, default="valid",
                        choices=["train", "valid", "test"],
                        help="Which split to evaluate")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pth)")

    # parser.add_argument("--image_size", type=int, default=256,
    #                     help="Resize image to image_size x image_size")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    # parser.add_argument("--download", action="store_true",
    #                     help="Download dataset if needed")

    return parser.parse_args()


def build_model(model_name: str) -> nn.Module:
    if model_name == "unet":
        model = UNet(
            in_channels=3,
            n_classes=1,
        )
    elif model_name == "resnet34_unet":
        model = ResNet34_UNet(
            in_channels=3,
            out_channels=1
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device,thr = 0.5):
    model.eval()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    for images, mask in loader:
        images = images.to(device)
        masks = mask.to(device)

        logits = model(images)               # [B,2,H,W]
        loss = criterion(logits, masks)
        dice = mean_dice_from_sigmoid_logits(logits, masks,threshold=thr)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        dice_meter.update(dice, batch_size)

    return loss_meter.avg, dice_meter.avg

@torch.no_grad()
def find_best_threshold(model, loader, device,trys=100):
    thresholds = np.array([i for i in range(1,trys)])/trys
    print(f"thresholds={thresholds}")
    best_dice = 0.0
    best_threshold = None
    for temp in thresholds:
        dice_meter = AverageMeter()
        for images, mask in loader:
            images = images.to(device)
            masks = mask.to(device) 

            logits = model(images)         
            dice = mean_dice_from_sigmoid_logits(logits, masks,threshold=temp)

            batch_size = images.size(0)
            dice_meter.update(dice, batch_size)
        if dice_meter.avg > best_dice:
            best_dice = dice_meter.avg
            best_threshold = temp

    return best_threshold

def main():
    args = get_args()

    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switch to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if args.model == 'unet':
        img_size = 388
        test_txt_name = 'test_unet.txt'
    else:
        img_size = 224
        test_txt_name = 'test_res_unet.txt'

    # image_size = (args.image_size, args.image_size)

    # train_dataset, valid_dataset, test_dataset = get_oxford_pet_datasets(
    #     root=args.data_root,
    #     split_dir=args.split_dir,
    #     image_size=image_size,
    #     train_augment=False,
    #     download=args.download,
    #     test_type=args.test_type
    # )
    image_dir = '../dataset/oxford-iiit-pet/images'
    mask_dir = '../dataset/oxford-iiit-pet/annotations/trimaps'
    txt_dir = '../dataset/splits'

    if args.eval_split == "train":
        eval_dataset = OxfordPetDataset(
                            image_dir=image_dir,
                            mask_dir=mask_dir,
                            txt_path=os.path.join(txt_dir,'train.txt'),
                            img_size=img_size,
                            mode="train",
                            model= args.model,
                            filter_small_mask=True,
                            min_fg_ratio=0.05
                        )
    elif args.eval_split == "valid":
        eval_dataset = OxfordPetDataset(
                        image_dir=image_dir,
                        mask_dir=mask_dir,
                        txt_path=os.path.join(txt_dir,'val.txt'),
                        img_size=img_size,
                        mode="val",
                        model= args.model
                    )
    elif args.eval_split == "test":
        eval_dataset = OxfordPetDataset(
                        image_dir=image_dir,
                        mask_dir=mask_dir,
                        txt_path=os.path.join(txt_dir,test_txt_name),
                        img_size=img_size,
                        mode="test",
                        model= args.model
                    )
    else:
        raise ValueError(f"Unsupported eval_split: {args.eval_split}")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = build_model(args.model).to(device)
    print_model_summary(model, model_name=args.model)

    _, _, checkpoint = load_checkpoint(
        checkpoint_path=os.path.join('../saved_models',args.checkpoint),
        model=model,
        optimizer=None,
        device=device
    )

    criterion = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)
    # criterion = nn.CrossEntropyLoss()

    print(f"Evaluate split : {args.eval_split}")
    print(f"Dataset size   : {len(eval_dataset)}")
    print(f"Checkpoint     : {args.checkpoint}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if "best_val_dice" in checkpoint:
        print(f"Checkpoint best_val_dice: {checkpoint['best_val_dice']:.4f}")
    print(f"Device         : {device}")
    print("=" * 60)

    eval_loss, eval_dice = evaluate(
        model=model,
        loader=eval_loader,
        criterion=criterion,
        device=device
    )

    print(f"Evaluation Loss : {eval_loss:.4f}")
    print(f"Evaluation Dice : {eval_dice:.4f}")

    thr = find_best_threshold(model, eval_loader, device,trys=100)
    print(thr)


if __name__ == "__main__":
    main()
    # python3 evaluate.py --data_root ../dataset/oxford-iiit-pet --split_dir ../dataset/splits --model resnet34_unet --eval_split valid --checkpoint res_unet/best.pth --batch_size 64 --device cuda:0