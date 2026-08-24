# Semantic Segmentation on Oxford-IIIT Pet

A PyTorch project for binary pet segmentation on the **Oxford-IIIT Pet Dataset**, comparing the original U-Net design with a ResNet34 encoder and U-Net-style decoder.

The complete workflow—data loading, augmentation, training, evaluation, checkpointing, and CSV inference—is exposed through small command-line programs in `src/`.

## Models

| Model | Encoder | Decoder | Intended comparison |
|---|---|---|---|
| U-Net | Original contracting path | Symmetric expanding path with skip connections | Train a segmentation network from scratch |
| ResNet34 U-Net | ResNet34 implemented from scratch | Bilinear upsampling, convolution blocks, and CBAM | Compare a residual encoder with the original U-Net |

## Quick start

```bash
git clone https://github.com/junyuan881/semantic-segmentation.git
cd semantic-segmentation

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Download the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) and arrange it as follows:

```text
dataset/
├── oxford-iiit-pet/
│   ├── images/               # *.jpg
│   └── annotations/
│       └── trimaps/          # *.png
└── splits/
    ├── train.txt
    ├── val.txt
    ├── test_unet.txt
    └── test_res_unet.txt
```

The split files are already included in this repository.

## Train

Run the commands from `src/` so the relative dataset and output paths match the project defaults.

```bash
cd src

# Original U-Net
python train.py \
  --data_root ../dataset/oxford-iiit-pet \
  --split_dir ../dataset/splits \
  --model unet \
  --epochs 150 \
  --batch_size 10 \
  --use_scheduler \
  --min_lr 1e-6 \
  --save_name unet

# ResNet34 U-Net
python train.py \
  --data_root ../dataset/oxford-iiit-pet \
  --split_dir ../dataset/splits \
  --model resnet34_unet \
  --epochs 150 \
  --batch_size 32 \
  --use_scheduler \
  --min_lr 5e-7 \
  --save_name res_unet
```

Checkpoints and training history are saved under `saved_models/<save_name>/`.

## Evaluate

```bash
# U-Net validation Dice score
python evaluate.py \
  --data_root ../dataset/oxford-iiit-pet \
  --split_dir ../dataset/splits \
  --model unet \
  --eval_split valid \
  --checkpoint ../saved_models/unet/best.pth

# ResNet34 U-Net validation Dice score
python evaluate.py \
  --data_root ../dataset/oxford-iiit-pet \
  --split_dir ../dataset/splits \
  --model resnet34_unet \
  --eval_split valid \
  --checkpoint ../saved_models/res_unet/best.pth
```

## Generate predictions

```bash
python inference.py \
  --data_root ../dataset/oxford-iiit-pet \
  --split_dir ../dataset/splits \
  --model unet \
  --checkpoint ../saved_models/unet/best.pth \
  --output_csv ../unet_predictions.csv
```

Replace the model and checkpoint with `resnet34_unet` and `../saved_models/res_unet/best.pth` to run the ResNet34 variant.

## Training design

- Trimap label `1` is treated as foreground; labels `2` and `3` are treated as background.
- Training examples with less than 5% foreground area are filtered out.
- Augmentation includes horizontal flips, color jitter, Gaussian blur, and rotations up to 15 degrees.
- Images are normalized with ImageNet statistics.
- The training objective is `0.3 × BCEWithLogitsLoss + 0.7 × DiceLoss`.
- Adam is used with weight decay, and `ReduceLROnPlateau` is available through `--use_scheduler`.
- The best and last checkpoints are saved separately for reproducible evaluation.

## Repository structure

```text
.
├── dataset/splits/          # Reproducible train/validation/test splits
├── src/
│   ├── models/
│   │   ├── unet.py
│   │   └── resnet34_unet.py
│   ├── oxford_pet.py        # Dataset and preprocessing pipeline
│   ├── train.py             # Training CLI
│   ├── evaluate.py          # Dice-score evaluation
│   ├── inference.py         # Test prediction and CSV export
│   └── EDA.ipynb            # Dataset exploration
├── requirements.txt
└── summary.txt              # Design and experiment notes
```

Run `python train.py --help`, `python evaluate.py --help`, or `python inference.py --help` for the complete set of options, including device selection and data-loader settings.

## Author

[junyuan881](https://github.com/junyuan881)

