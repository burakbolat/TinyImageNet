import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch


class TrainTinyImageNet(Dataset):
    """ Train dataset class. This is different from Test and Train since folder organization is so.
    Classes are sorted in numerical order and labels assigned wrt order of the folders in the directory.
    For instance, n01443537 is the 0th class since it is the smallest object id.
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.folders_sorted = sorted(os.listdir(img_dir))
        self.transform = transform
        self.image_per_class = 500
        self.num_of_classes = 200

    def __len__(self):
        return self.image_per_class * self.num_of_classes

    def __getitem__(self, idx):
        img_external_idx = idx // self.image_per_class
        img_internal_idx = idx % self.image_per_class
        img_folder = self.folders_sorted[img_external_idx]
        
        img_path = os.path.join(self.img_dir, 
                                img_folder, 
                                "images", 
                                "{}_{}.JPEG".format(img_folder, img_internal_idx))

        image = read_image(img_path)/255.0

        if image.size() == torch.Size([1, 64, 64]):
            image = image.repeat(3, 1, 1)
        

        if self.transform:
            image = self.transform(image)

        label = img_external_idx
        return image, label

if __name__=="__main__":
    dataset = TrainTinyImageNet("tiny-imagenet-200/train")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)