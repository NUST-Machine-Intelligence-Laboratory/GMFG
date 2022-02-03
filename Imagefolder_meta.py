from  torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import random
import numpy as np

random.seed(0)
np.random.seed(0)

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None, number = None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            if number!= None:
                if len(fnames) > number:
                    fnames = random.sample(fnames, number)
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root_train, root_meta, loader, extensions=None, transform=None, transform_meta=None, target_transform=None, is_valid_file=None, cached=False, number=None, plus= False):
        super(DatasetFolder, self).__init__(root=root_train)
        self.transform = transform
        self.target_transform = target_transform
        if transform_meta == None:
            self.transform_meta = transform
        else:
            self.transform_meta = transform_meta
        self.cached=cached
        self.root_train = root_train
        self.root_meta = root_meta
        classes_train, class_to_idx_train = self._find_classes(self.root_train)
        samples_train = make_dataset(self.root_train, class_to_idx_train, extensions, is_valid_file, number)

        classes_meta, class_to_idx_meta = self._find_classes(self.root_meta)
        samples_meta = make_dataset(self.root_meta, class_to_idx_meta, extensions, is_valid_file)

        if len(samples_train) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root_train + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.classes = classes_train

        self.samples_meta = samples_meta
        if plus:
            self.samples_train = samples_train + samples_meta
        else:
            self.samples_train = samples_train

        print('--------------------------------')
        print('preparing dataset')
        if number != None:
            print('Using part of images, number of each class: ',number)
        else:
            print('Using all images')
        if cached == True:
            print('load all images once')
            self.images_train=[]
            for sample in self.samples_train:
                path, target = sample
                # image = self.loader(path)
                self.images_train.append(self.loader(path))

        self.images_meta = []
        for sample in self.samples_meta:
            path, target = sample
            # image = self.loader(path)
            self.images_meta.append(self.loader(path))
        print('number of meta images:',len(self.samples_meta))
        print('number of training images:', len(self.samples_train))
        print('--------------------------------')

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path_train, target_train = self.samples_train[index]

        if self.cached == False:
            # print(path)
            sample_train = self.loader(path_train)
        else:
            sample_train = self.images_train[index]
        if self.transform is not None:
            sample_train = self.transform(sample_train)
        if self.target_transform is not None:
            target_train = self.target_transform(target_train)

        # index_meta = (index + np.random.randint(0, len(self.samples_train))) % len(self.samples_meta)
        index_meta = np.random.randint(0, len(self.samples_meta))
        path_meta, target_meta = self.samples_meta[index_meta]
        sample_meta = self.images_meta[index_meta]
        if self.transform_meta is not None:
            # sample_meta_t = self.transform_meta(sample_meta)
            sample_meta = self.transform(sample_meta)
        if self.target_transform is not None:
            target_meta = self.target_transform(target_meta)

        return sample_train, target_train, index, path_train, sample_meta, target_meta

    def __len__(self):
        return len(self.samples_train)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

class Imagefolder_meta(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root_train, root_meta, transform=None,transform_meta=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, cached=False, number=None, plus=False):
        super(Imagefolder_meta, self).__init__(root_train, root_meta, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          transform_meta=transform_meta,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                                   cached=cached,
                                                   number=number,
                                               plus=plus)
