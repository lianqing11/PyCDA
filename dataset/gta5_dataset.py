import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from .utils import RandomResizedCrop

class GTA5DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(1024, 512), transform = None, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.transform = transform
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = np.asarray(label)
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy)
        name = datafiles["name"]

        # resize
        if self.transform:
            #print(type(image), type(label))
            image, label = self.transform(image, label)

        #print(np.unique(label))
        # re-assign labels to match the format of Cityscapes

        return image, label, name


