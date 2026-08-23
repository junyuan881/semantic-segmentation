import os
import json
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from oxford_pet import get_dataloader
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import (
    set_seed,
    AverageMeter,
    mean_dice_from_logits,
    init_history,
    update_history,
    save_checkpoint,
    ensure_dir,
    print_model_summary,
    BCEDiceLoss,
    init_weights_he,
    mean_dice_from_sigmoid_logits,
)


def get_args():
    parser = argparse.ArgumentParser(description="Train binary semantic segmentation models")

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

    # parser.add_argument("--image_size", type=int, default=256,
    #                     help="Resize image to image_size x image_size")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="../saved_models")
    parser.add_argument("--save_name", type=str, default=None,
                        help="Checkpoint base name. If None, auto use model name")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    # parser.add_argument("--download", action="store_true",
    #                     help="Download dataset if needed")
    parser.add_argument("--use_scheduler", action="store_true",
                    help="Use learning rate scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=3,
                        help="Scheduler patience")
    parser.add_argument("--scheduler_factor", type=float, default=0.5,
                        help="LR decay factor")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate")
    parser.add_argument("--bce_weight", type=float, default=0.3,
                        help="BCE loss weight")
    parser.add_argument("--dice_weight", type=float, default=0.7,
                        help="DICE loss weight")

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
            out_channels=1,
            use_cbam = True
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    for images,masks in loader:
        images = images.to(device)   # [B,3,H,W]
        masks = masks.to(device)     # [B,H,W]

        optimizer.zero_grad()

        logits = model(images)               # [B,2,H,W]
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        dice = mean_dice_from_sigmoid_logits(logits, masks)

        loss_meter.update(loss.item(), batch_size)
        dice_meter.update(dice, batch_size)

    return loss_meter.avg, dice_meter.avg


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    for images, mask in loader:
        images = images.to(device)
        masks = mask.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        batch_size = images.size(0)
        dice = mean_dice_from_sigmoid_logits(logits, masks)

        loss_meter.update(loss.item(), batch_size)
        dice_meter.update(dice, batch_size)

    return loss_meter.avg, dice_meter.avg


def main():
    args = get_args()
    print(f"""Train parameters: \n\
            device: {args.device} \n\
            model: {args.model} \n\
            data_root: {args.data_root} \n\
            batch_size: {args.batch_size} \n\
            learning rate: {args.lr} \n\
            weight_decay: {args.weight_decay} \n\
            bce_weight: {args.bce_weight} \n\
            dice_weight: {args.dice_weight} \n\
            use_scheduler: {args.use_scheduler} \n\
            scheduler_factor: {args.scheduler_factor} \n\
            scheduler_patience: {args.scheduler_patience} \n\
            save_name: {args.save_name}\n""")

    # set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switch to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.model=='unet':
        image_size = 388
    else:
        image_size = 224

    train_loader, val_loader, _ = get_dataloader(args.data_root+'/images',args.data_root+'/annotations/trimaps',args.split_dir,model = args.model,batch_size=args.batch_size,img_size=image_size)

    model = build_model(args.model).to(device)
    model.apply(init_weights_he)
    print_model_summary(model, model_name=args.model)

    criterion = BCEDiceLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        # betas=(0.99,0.999) # betas=(0.9, 0.999)
    )
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",                 # 因為你監控的是 val_dice，越大越好
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_lr
        )
    history = init_history()
    best_val_dice = -1.0

    save_name = args.save_name if args.save_name is not None else args.model
    model_save_dir = os.path.join(args.save_dir, save_name)# +'_'+str(datetime.datetime.now().date()))
    ensure_dir(model_save_dir)

    best_ckpt_path = os.path.join(model_save_dir, "best.pth")
    last_ckpt_path = os.path.join(model_save_dir, "last.pth")
    history_path = os.path.join(model_save_dir, "history.json")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_dice = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )
        if scheduler is not None:
            scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]["lr"]

        history = update_history(
            history,
            train_loss=train_loss,
            train_dice=train_dice,
            val_loss=val_loss,
            val_dice=val_dice
        )

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"LR: {current_lr:.6e} | "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

        save_checkpoint(
            save_path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_dice=best_val_dice,
            history=history
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint(
                save_path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_dice=best_val_dice,
                history=history
            )
            print(f"New best model saved. Best Val Dice: {best_val_dice:.4f}")

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("=" * 60)
    print(f"Training finished.")
    print(f"Best Val Dice: {best_val_dice:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()

    # python3 train.py --data_root ../dataset/oxford-iiit-pet --split_dir ../dataset/splits --model unet --epochs 150 --batch_size 10 --device cuda:1 --use_scheduler --scheduler_patience 3 --scheduler_factor 0.5 --min_lr 1e-6 --save_name unet
    # python3 train.py --data_root ../dataset/oxford-iiit-pet --split_dir ../dataset/splits --model resnet34_unet --epochs 150 --batch_size 32 --device cuda:0 --use_scheduler --scheduler_patience 3 --scheduler_factor 0.5 --min_lr 5e-7 --save_name res_unet