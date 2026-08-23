# Usage

- Name: 謝濬遠
- Student ID: 114024511

## Data prepare

**STEP 1**\
install the requirements
```bash
pip install -r requirements.txt
```

**STEP 2**\
download dataset put in oxford-iiit-pet
```
├── dataset/
│   ├── oxford-iiit-pet/
│   │   ├── annotations.tar.gz/
│   │   └── images.tar.gz/
│   └── splits/
│       ├── train.txt/
│       ├── test_res_unet.txt/
│       ├── val.txt/
│       └── test_unet.txt/
├── src/
│   ├── models/
│   │   ├── unet.py
│   │   └── resnet34_unet.py
│   ├── oxford_pet.py
│   ├── utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── saved_models/
├── submissions/
└── requirements.txt
```
and then unzip it(run the following command below the oxford-iiit-pet)
```bash
tar -xzvf images.tar.gz
tar -xzvf annotations.tar.gz
```

## Training
the following command are all use under src
unet: 
```bash
python3 train.py --data_root ../dataset/oxford-iiit-pet --split_dir ../dataset/splits --model unet --epochs 150 --batch_size 10 --device cuda --use_scheduler --scheduler_patience 3 --scheduler_factor 0.5 --min_lr 1e-6 --save_name unet
```

resnet34_unet: 
```bash
python3 train.py --data_root ../dataset/oxford-iiit-pet --split_dir ../dataset/splits --model resnet34_unet --epochs 150 --batch_size 32 --device cuda --use_scheduler --scheduler_patience 3 --scheduler_factor 0.5 --min_lr 5e-7 --save_name res_unet
```

## Evaluate
```bash
python3 evaluate.py --data_root ../dataset/oxford-iiit-pet --split_dir ../dataset/splits --model unet --eval_split valid --checkpoint unet/best.pth --batch_size 16 --device cuda
```

## Inference
unet:
```bash
python3 inference.py --data_root ../dataset/oxford-iiit-pet --split_dir ../dataset/splits --model unet --checkpoint ../saved_models/unet/best.pth --batch_size 8 --device cuda --output_csv ../submissions/unet_submission.csv
```

resnet34_unet:
```bash
python3 inference.py --data_root ../dataset/oxford-iiit-pet --split_dir ../dataset/splits --model resnet34_unet --checkpoint ../saved_models/res_unet/best.pth --batch_size 8 --device cuda --output_csv ../submissions/res_unet_submission.csv
```
