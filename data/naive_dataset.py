import os

import torch
from PIL import Image

from data import BaseDataset
from data.base_dataset import get_transform
from utils import multi_class_remap


class NaiveDataset(BaseDataset):
    """ A dataset class for labeled image dataset.

        The file structure should be:
        - data_root
            - 0
                - image.png (original image)
                - label.png (ground truth)
                - mask.png (used to ignore unwanted pixels)
            - 1
            - 2
            ...
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_mask', action='store_true', help='whether the dataset has mask')
        parser.add_argument('--full_mask', action='store_true', help='generate a mask with all 1s')
        return parser

    def __init__(self, opt):
        """ Initialize this dataset class.

        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.opt = opt
        self.len = len(os.listdir(self.opt.data_dirname))

    def __getitem__(self, index):
        """ Return a data dict and its metadata information.

        :param index: an integer for data indexing
        :return a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        raw_transform, label_transform = get_transform(self.opt)

        original_path = os.path.join(self.opt.data_dirname, str(index), 'image.png')
        original = Image.open(original_path).convert('RGB')
        original = raw_transform(original)

        label_path = os.path.join(self.opt.data_dirname, str(index), 'label.png')
        label = Image.open(label_path).convert('L')
        label = label_transform(label)


        if self.opt.output_nc > 1:
            label = (label * 255).squeeze().long()
            label = multi_class_remap(label, self.opt.multi_class_label_remap)

        if self.opt.full_mask:
            mask = torch.ones_like(label)
            return {'image_original': original, 'label': label, 'source_path': original_path, 'mask': mask}

        if not self.opt.no_mask:
            mask_path = os.path.join(self.opt.data_dirname, str(index), 'mask.png')
            mask = Image.open(mask_path).convert('L')
            mask = label_transform(mask)
            return {'image_original': original, 'label': label, 'source_path': original_path, 'mask': mask}

        return {'image_original': original, 'label': label, 'source_path': original_path}

    def __len__(self):
        """ Return the total number of images in the dataset."""
        return self.len
