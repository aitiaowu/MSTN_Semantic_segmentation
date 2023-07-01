import os
import os.path as osp
from PIL import Image
import numpy as np
import torch
from torch.utils import data


class vesselDataset(data.Dataset):
    def __init__(self, root, list_path, transform=None):
        self.root = root
        self.list_path = list_path  # list of image names
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.ignore_label = 255
        #self.id_to_trainid = {1: 0, 2: 1, 3: 2}
        self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3}
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        img_name = name.replace('label', 'image')
        
        image = Image.open(osp.join(self.root, "images/%s" % img_name)).convert('RGB')
        label = Image.open(osp.join(self.root, "labeled_data/%s" % name)) #.convert('L')
        
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        #print(image.shape,label.shape)

        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # convert to BGR
        image = image.transpose((2, 0, 1))
        #print(image.shape,label.shape)

        if self.transform is not None:
            augmentations = self.transform(image=image, label_copy=label_copy)
            image = augmentations["image"]
            label_copy = augmentations["label_copy"]

        return image.copy(), label_copy.copy(), name


