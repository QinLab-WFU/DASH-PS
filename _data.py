import configparser
import os
import platform

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MyDataset(Dataset):
    """
    Common dataset for DeepHashing.

    Args
        dataset(str): Dataset name.
        img_dir(str): Directory of image files.
        txt_dir(str): Directory of txt file containing image file names & image labels.
        transform(callable, optional): Transform images.
    """

    def __init__(self, root, dataset, usage, transform=None):
        assert dataset in ["cifar", "flickr", "coco", "nuswide"]
        self.name = dataset

        assert usage in ["train", "query", "dbase"]

        if not os.path.exists(root):
            print(f"root not exists: {root}")
            root = os.path.dirname(__file__) + "/_datasets"
            print(f"root will use: {root}")

        xxx_dir = os.path.join(root, f"{dataset}")
        img_dir = f"{xxx_dir}/images/"
        # img_loc = os.path.join(img_dir, "images_location.txt")
        ini_loc = os.path.join(img_dir, "images_location.ini")
        if os.path.exists(ini_loc):
            # self.img_dir = open(img_loc, "r").readline()
            config = configparser.ConfigParser()
            config.read(ini_loc)
            self.img_dir = config["DEFAULT"][platform.system()]
        else:
            self.img_dir = img_dir

        self.transform = build_default_trans(usage) if transform is None else transform

        # Read files
        self.data = [
            (x.split()[0], np.array(x.split()[1:], dtype=np.float32))
            for x in open(os.path.join(xxx_dir, f"{usage}.txt"), "r").readlines()
        ]

    def __getitem__(self, index):
        file_name, label = self.data[index]
        with open(os.path.join(self.img_dir, file_name), "rb") as fp:
            img = Image.open(fp).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        return torch.from_numpy(np.vstack([x[1] for x in self.data]))


def get_class_num(dataset):
    r = {"cifar": 10, "flickr": 38, "nuswide": 21, "coco": 80}[dataset]
    return r


def get_topk(dataset):
    r = {"cifar": -1, "flickr": -1, "nuswide": 5000, "coco": -1}[dataset]
    return r


def build_default_trans(usage):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if usage == "train":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transform


def build_loader(root, dataset, usage, transform, **kwargs):
    dset = MyDataset(root, dataset, usage, transform)
    print(f"{usage} set length: {len(dset)}")
    loader = DataLoader(dset, shuffle=usage == "train", **kwargs)
    return loader


def build_loaders(root, dataset, trans_train=None, trans_test=None, **kwargs):
    loaders = []
    for usage in ["train", "query", "dbase"]:
        loaders.append(
            build_loader(
                root,
                dataset,
                usage,
                trans_train if usage == "train" else trans_test,
                **kwargs,
            )
        )

    return loaders


if __name__ == "__main__":
    _, query_loader, _ = build_loaders("/root/autodl-tmp/ASL/_datasets/", "coco", batch_size=2, num_workers=4)
    print("topk", get_topk("coco"))
    print("num_classes", get_class_num("coco"))
    for x in query_loader:
        print(x[0], x[1])
        break
