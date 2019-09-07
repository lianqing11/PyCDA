import os
import os.path as osp
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from .utils import RandomResizedCrop


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.float32)


class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(1024, 512), ignore_label=255, transform = None, set='val', dataset_info = None, need_label = False):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        self.transform = transform
        self.need_label = need_label
        with open('./dataset/cityscapes_list/info.json', 'r') as fp:
            dataset_info = json.load(fp)
        self.mapping = np.array(dataset_info['label2train'], dtype=np.int)

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            name = name.split(" ")[0]
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            lbl_name = "_".join(name.split("_")[:-1]) + "_gtFine_labelIds.png"
            lbl_file = osp.join(self.root, "gtFine/%s/%s"%(self.set, lbl_name))
            self.files.append({
                "img": img_file,
                "name": name,
                "label": lbl_file
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        if self.need_label == True:

            label = Image.open(datafiles["label"])
            label = label_mapping(np.asarray(label), self.mapping)
            label = Image.fromarray(label)
            if self.transform:
                image, label = self.transform(image, label)
        else:
            if self.transform:
                image = self.transform(image)


        if self.need_label == True:

            return image, label, name
        else:
            return image, name



class fake_cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(1024, 512), ignore_label=255, transform = None, set='val', dataset_info = None, need_label = False):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        self.transform = transform
        self.need_label = True
        with open('./dataset/cityscapes_list/info.json', 'r') as fp:
           dataset_info = json.load(fp)
        self.mapping = np.array(dataset_info['label2train'], dtype=np.int)

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            name = name.split(" ")[0]
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            lbl_name = "_".join(name.split("_")[:-1]) + "_gtFine_labelIds.png"
            lbl_file = osp.join(self.root, "gtFine/%s/%s"%(self.set, lbl_name))
            self.files.append({
                "img": img_file,
                "name": name,
                "label": lbl_file
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        if self.need_label == True:

            label = np.zeros((560, 480), dtype=np.uint8)
            label = Image.fromarray(label)
            if self.transform:
                image, label = self.transform(image, label)

        return image, label, name

