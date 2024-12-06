import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, root: str = r"D:\HardExudates", train: bool = True, transforms=None):
        super(SegmentationDataset, self).__init__()
        self.flag = "train" if train else "test"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        self.transforms = transforms

        # Load image names
        img_names = [i for i in os.listdir(os.path.join(data_root, "images"))]
        
        # Create full paths for images and masks
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.masks = [os.path.join(data_root, "1st_manual", i.replace(".jpg", "_EX.tif")) for i in img_names]

        # Check if all masks exist
        for mask in self.masks:
            if not os.path.exists(mask):
                raise FileNotFoundError(f"Mask file '{mask}' does not exist.")

    def __getitem__(self, item):
        image_path = self.img_list[item]
        mask_path = self.masks[item]
        
        # Load the image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Convert mask to binary (0, 1)
        mask = np.array(mask) // 255
        mask = Image.fromarray(mask)

        # Apply transformations if any
        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
