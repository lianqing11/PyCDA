import random
import math
import numpy as np
import numbers
import collections
from PIL import Image, ImageOps, ImageFilter

import torch


class Compose(object):
    """
    Composes several segsegtransforms together.

    Args:
        segtransforms (List[Transform]): list of segtransforms to compose.

    Example:
        segtransforms.Compose([
            segtransforms.CenterCrop(10),
            segtransforms.ToTensor()])
    """
    def __init__(self, segtransforms):
        self.segtransforms = segtransforms

    def __call__(self, image, label):
        for t in self.segtransforms:
            image, label = t(image, label)
        return image, label


class ToTensor(object):
    # Converts a PIL Image or numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image, label):
        if isinstance(image, Image.Image) and isinstance(label, Image.Image):
            image = np.asarray(image)
            label = np.asarray(label)
        elif not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransforms.ToTensor() only handle PIL Image and np.ndarray"
                                "[eg: data readed by PIL.Image.open()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransforms.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise (RuntimeError("segtransforms.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return image, label


class Normalize(object):
    """
    Given mean and std of each channel
    Will normalize each channel of the torch.*Tensor (C*H*W), i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        assert image.size(0) == len(self.mean)
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image, label


class Resize(object):
    """
    Resize the input PIL Image to the given size.
    'size' is a 2-element tuple or list in the order of (h, w)
    """
    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, image, label):
        image = image.resize(self.size[::-1], Image.BILINEAR)
        label = label.resize(self.size[::-1], Image.NEAREST)
        return image, label


class RandScale(object):
    """
    Randomly resize image & label with scale factor in [scale_min, scale_max]
    """
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransforms.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransforms.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, label=None):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_w = temp_scale * temp_aspect_ratio
        scale_factor_h = temp_scale / temp_aspect_ratio
        w, h = image.size
        new_w = int(w * scale_factor_w)
        new_h = int(h * scale_factor_h)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        if label is not None:
            label = label.resize((new_w, new_h), Image.NEAREST)
            return image, label
        else:
            return image

class Crop(object):
    """Crops the given PIL Image.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label=None):
        w, h = image.size
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransforms.Crop() need padding while padding argument is None\n"))
            border = (pad_w_half, pad_h_half, pad_w - pad_w_half, pad_h - pad_h_half)
            image = ImageOps.expand(image, border=border, fill=tuple([int(item) for item in self.padding]))
            if label is not None:
                label = ImageOps.expand(label, border=border, fill=self.ignore_label)
        w, h = image.size
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = (h - self.crop_h) / 2
            w_off = (w - self.crop_w) / 2
        image = image.crop((w_off, h_off, w_off+self.crop_w, h_off+self.crop_h))
        if label is not None:
            label = label.crop((w_off, h_off, w_off+self.crop_w, h_off+self.crop_h))
            return image, label
        else:
            return image

class RandRotate(object):
    """
    Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    """
    def __init__(self, rotate, padding, ignore_label=255):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransforms.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label

    def __call__(self, image, label=None):
        angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
        mask = Image.new('L', image.size, 255)
        mask = mask.rotate(angle, Image.NEAREST)
        image_bg = Image.new(image.mode, image.size, tuple([int(item) for item in self.padding]))
        if label is not None:
            label_bg = Image.new(label.mode, label.size, self.ignore_label)
        image_bg.paste(image.rotate(angle, Image.BILINEAR), mask)
        if label is not None:
            label_bg.paste(label.rotate(angle, Image.NEAREST), mask)
            return image_bg, label_bg
        else:
            return image_bg

class RandomHorizontalFlip(object):
    def __call__(self, image, label=None):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if label is not None:
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if label is not None:
            return image, label
        else:
            return image

class RandomVerticalFlip(object):
    def __call__(self, image, label):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
        return image, label


class RandomGaussianBlur(object):
    def __init__(self, radius=2):
        self.radius=2
    def __call__(self, image, label=None):
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(self.radius))
        if label is not None: 
            return image, label
        else: 
            return image

class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image, label):
        r, g, b = image.split()
        image = Image.merge('RGB', (b, g, r))
        return image, label
