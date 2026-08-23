import os
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random

def get_filenames(txt_path):
    filenames = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                filenames.append(name)

    return filenames


class OxfordPetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, txt_path, img_size=388, mode="train",model='unet',filter_small_mask=False,min_fg_ratio=0.05):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        # self.pad = pad
        self.mode = mode
        self.model = model

        # self.filenames = [f.split('.')[0] for f in os.listdir(image_dir)]
        self.filenames = get_filenames(txt_path)
        self.filenames.sort()
        self.filter_small_mask = filter_small_mask
        self.min_fg_ratio = min_fg_ratio
        if self.mode == "train" and self.filter_small_mask:
            self.filenames = self._filter_filenames_by_mask_ratio(self.filenames , self.min_fg_ratio)
        
        if mode == 'eval':
            self.img_size = None
        # if mode == 'test':
        #     print(self.filenames)
        
    def _filter_filenames_by_mask_ratio(self, filenames, min_fg_ratio):
        filtered_filenames = []
        removed_count = 0

        for filename in filenames:
            mask_path = os.path.join(self.mask_dir, filename + ".png")
            mask = Image.open(mask_path)
            mask_np = np.array(mask, dtype=np.uint8)
            mask_np[mask_np == 2] = 0
            mask_np[mask_np == 1] = 1
            mask_np[mask_np == 3] = 0

            fg_ratio = mask_np.mean()   # 因為 mask 是 0/1，所以 mean = 前景比例

            if fg_ratio >= min_fg_ratio:
                filtered_filenames.append(filename)
            else:
                removed_count += 1

        print(
            f"[Mask Filter] split={self.mode}, "
            f"keep={len(filtered_filenames)}, remove={removed_count}, "
            f"min_fg_ratio={min_fg_ratio}"
        )
        return filtered_filenames
    def _appearance_augment(self, image: Image.Image) -> Image.Image:
        """
        只對 image 做外觀變化，不改 mask
        """

        # 1) brightness
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)

        # 2) contrast
        if random.random() < 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)

        # 3) saturation
        if random.random() < 0.3:
            saturation_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_saturation(image, saturation_factor)

        # 4) hue
        if random.random() < 0.3:
            hue_factor = random.uniform(-0.05, 0.05)
            image = TF.adjust_hue(image, hue_factor)

        # # 5) gaussian blur
        if random.random() < 0.2:
            radius = random.uniform(0.1, 1.2)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        return image

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        img_path = os.path.join(self.image_dir, name + ".jpg")
        mask_path = os.path.join(self.mask_dir, name + ".png")
        # print(mask_path)
        if self.model == 'unet':
            image = Image.open(img_path).convert('RGB')
            pad = 92
        else:
            image = Image.open(img_path).convert('RGB')
            pad = None
        mask = Image.open(mask_path)
        origin_shape = image.size
        # shape = image.size

        # ---------- Resize ----------
        if self.img_size is not None:
            image = image.resize((self.img_size, self.img_size))
            mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        # ---------- Data Augmentation ----------
        if self.mode == "train":
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # if random.random() > 0.5:
            #     image = TF.vflip(image)
            #     mask = TF.vflip(mask)
            image = self._appearance_augment(image)

            # random rotation
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # ---------- To Tensor ----------
        image = TF.to_tensor(image) 
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if pad is not None:
            image = F.pad(image, (pad, pad, pad, pad), mode="reflect")
        else:
            pass
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0)

        # ---------- Label 處理 ----------
        # 原始: 1=pet, 2=background, 3=boundary
        mask[mask == 2] = 0
        mask[mask == 1] = 1
        mask[mask == 3] = 0
        
        if self.mode == 'test' or self.mode == 'eval':
            return image, mask, origin_shape, name
        else:
            return image, mask #if self.mode == 'train' or self.mode == 'val' else image, mask, shape, name
    

def get_dataloader(image_dir, mask_dir,txt_dir, model = 'unet', batch_size=8, img_size=388):
    if model == 'unet':
        test_txt_name = 'test_unet.txt'
    else:
        test_txt_name = 'test_res_unet.txt'

    train_dataset = OxfordPetDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        txt_path=os.path.join(txt_dir,'train.txt'),
        img_size=img_size,
        mode="train",
        model= model,
        filter_small_mask=True,
        min_fg_ratio=0.05
    )
    val_dataset = OxfordPetDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        txt_path=os.path.join(txt_dir,'val.txt'),
        img_size=img_size,
        mode="val",
        model= model
    )
    test_dataset = OxfordPetDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        txt_path=os.path.join(txt_dir,test_txt_name),
        img_size=img_size,
        mode="test",
        model= model
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(val_dataset)}")
    print(f"test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    image_dir = '../dataset/oxford-iiit-pet/images'
    mask_dir = '../dataset/oxford-iiit-pet/annotations/trimaps'
    txt_dir = '../dataset/splits'
    train_loader, val_loader, test_loader = get_dataloader(image_dir,mask_dir,txt_dir)
    for image,masks in train_loader:
        print(image[0].size())
        break
