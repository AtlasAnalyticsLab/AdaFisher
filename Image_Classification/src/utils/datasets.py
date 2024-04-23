from contextlib import contextmanager
from pathlib import Path

import warnings
import tempfile
import shutil
import os
import torch
from torchvision.datasets.utils import check_integrity,\
    extract_archive, verify_str_arg, download_and_extract_archive
from torchvision.datasets.folder import ImageFolder


class TinyImageNet(ImageFolder):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    meta_file = 'wnids.txt'
    """`TinyImageNet
    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or
            ``val``.
        transform (callable, optional): A function/transform that  takes in an
            PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its
            path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, split='train', download=False, **kwargs):
        root = self.root = os.path.expanduser(root) + "/tiny-imagenet-200"
        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.root = root
        if download:
            # self.download()
            raise ValueError(
                "Downloading of TinyImageNet is not supported. " +
                "You must manually download the 'tiny-imagenet-200.zip' from" +
                f" {self.url} and extract the 'tiny-imagenet-200' folder " +
                "into the folder specified by 'root'. That is, once the" +
                "'tiny-imagenet-200' folder is extracted, specify the data " +
                "directory for this program as the path for to that folder")
        self.parse_archives()
        self.classes = self.load_meta_file()
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}
        super(TinyImageNet, self).__init__(self.split_folder, **kwargs)

    def _check_integrity(self):
        dirs = [d.name for d in Path(self.root).iterdir()]
        if 'train' not in dirs or 'test' not in dirs or 'val' not in dirs:
            return False
        if not (Path(self.root) / 'wnids.txt').exists():
            return False
        return True

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
        download_and_extract_archive(
            self.url, self.root,
            filename=self.filename, md5=None)

    def load_meta_file(self):
        if self._check_integrity():
            with (Path(self.root) / self.meta_file).open('r') as f:
                lines = [line.strip() for line in f.readlines()]
        return lines

    def parse_archives(self):
        if self._check_integrity():
            name = (Path(self.root) / 'train')
            if (name / 'images').exists():
                for c in name.iterdir():
                    os.remove(str(c / f'{c.name}_boxes.txt'))
                    for f in (c / 'images').iterdir():
                        shutil.move(str(f), c)
                    shutil.rmtree(str(c / 'images'))
            name = (Path(self.root) / 'val')
            if (name / 'images').exists():
                with (name / 'val_annotations.txt').open('r') as f:
                    for line in f.readlines():
                        line = line.replace('\t', ' ').strip().split(' ')
                        (name / line[1]).mkdir(exist_ok=True)
                        shutil.move(str(name / 'images' / line[0]),
                                    str(name / line[1]))
                shutil.rmtree(str(name / 'images'))
                os.remove(name / 'val_annotations.txt')

    @ property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)