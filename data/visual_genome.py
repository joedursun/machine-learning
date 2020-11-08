# Some of this code was copied from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import torch
from torch.utils.data import Dataset, DataLoader
import glob, os
from skimage import io, transform
import numpy as np

class VisualGenomeDataset(Dataset):
    """Visual Genome dataset."""

    def __init__(self, root_dir, transform=None, cache_dir=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_glob_str = os.path.join(root_dir, '*.jpg')
        image_names = glob.glob(self.image_glob_str)
        self.image_names = []
        if cache_dir:
            self.image_names = image_names

        num_images = len( image_names )
        assert num_images > 0, "No images found in root_dir"
        self.num_images = num_images
        self.transform = transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if len(self.image_names) > idx: # want to avoid index out of bounds error
            img_name = self.image_names[idx]
        else:
            img_name = glob.glob(self.image_glob_str)[idx]

        image = io.imread(img_name)
#         image = torch.from_numpy(image)

        if self.transform:
            try:
                image = self.transform(image)
            except (RuntimeError, TypeError, NameError) as err:
                print(err, img_name)
                return

        return image

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]

        return image
