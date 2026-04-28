# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Modified on torchvision code bases
# https://github.com/pytorch/vision
# --------------------------------------------------------'
from torchvision.datasets.vision import VisionDataset

from PIL import Image

import re
import os
import os.path
import random
import json
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


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
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            index_file: Optional[str] = None, 
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        if index_file is None:
            classes, class_to_idx = self._find_classes(self.root)
            samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
            if len(samples) == 0:
                msg = "Found 0 files in subfolders of: {}\n".format(self.root)
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                raise RuntimeError(msg)
        else:
            with open(index_file, mode="r", encoding="utf-8") as reader:
                classes = []
                index_data = {}
                for line in reader:
                    data = json.loads(line)
                    class_name = data["class"]
                    classes.append(class_name)
                    index_data[class_name] = data["files"]
                
                classes.sort()
                class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
                samples = []
                for class_name in index_data:
                    class_index = class_to_idx[class_name]
                    for each_file in index_data[class_name]:
                        samples.append(
                            (os.path.join(root, class_name, each_file), 
                            class_index)
                        )

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        print("Find %d classes and %d samples in root!" % (len(classes), len(samples)))

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                path, target = self.samples[index]
                sample = self.loader(path)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.samples[i][0]) for i in indices]
            else:
                return [self.samples[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.samples]
            else:
                return [x[0] for x in self.samples]


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
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
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            index_file: Optional[str] = None, 
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, index_file=index_file)
        self.imgs = self.samples

        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            while True:
                try:
                    path, target = self.samples[index]
                    sample = self.loader(path)  # sample is a PIL Image object

                    # --- START: Custom On-the-Fly Rotation Correction ---
                    # Use the new comprehensive parsing function for the filename
                    try:
                        parsed_data = parse_full_image_metadata(path)  # Pass full path for basename
                        station_name = parsed_data['station']
                        channel_int = parsed_data['channel']
                        rotate_based_on_path = SPECIAL_ROTATION_FOLDER in path
                    except ValueError as e:
                        # This catches truly malformed "normal" images that don't fit the expected patterns
                        print(f"Error parsing image {os.path.basename(path)}: {e}. Trying another random image.")
                        index = random.randint(0, len(self.samples) - 1)
                        continue  # Skip to the next image in the loop

                    # Determine if rotation is needed using the loaded station_dict
                    # get_rotation_flag will return False for "NO_ROTATION_STATION_L" or channel 0
                    needs_rotation = get_rotation_flag(str(channel_int), station_name, station_dict)
                    needs_rotation = (needs_rotation or rotate_based_on_path)

                    if needs_rotation:
                        # Apply 270 degrees anti-clockwise rotation (equivalent to 90 degrees clockwise)
                        sample = sample.transpose(Image.ROTATE_270)
                    # --- END: Custom On-the-Fly Rotation Correction ---

                    break  # Break from while True loop if image loaded and processed without error
                except Exception as e:
                    # Catch any other loading or general processing errors (e.g., corrupted image file)
                    print(f"General error loading or processing image {path}: {e}. Trying another random image.")
                    index = random.randint(0, len(self.samples) - 1)

            if self.transform is not None:
                sample = self.transform(sample)  # Apply other torchvision transforms
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target

        def __len__(self) -> int:
            return len(self.samples)

        def filenames(self, indices=[], basename=False):
            if indices:
                if basename:
                    return [os.path.basename(self.samples[i][0]) for i in indices]
                else:
                    return [self.samples[i][0] for i in indices]
            else:
                if basename:
                    return [os.path.basename(x[0]) for x in self.samples]
                else:
                    return [x[0] for x in self.samples]

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
SPECIAL_ROTATION_FOLDER = '/data1/beit2/data/guangzhou_simulation_large'

# --- START: Custom image metadata parsing and rotation logic ---
# (Keep your existing station_dict loading logic here)
try:
    # current_script_dir = os.path.dirname(os.path.abspath(__file__))
    station_rotation_json_path = '/home/suma/bin/try_detectron2/multipleGPUs/station_rotation.json'
    # os.path.join(current_script_dir, 'station_rotation.json')
    with open(station_rotation_json_path, 'r') as file:
        station_dict = json.load(file)
except FileNotFoundError:
    print(
        f"Warning: station_rotation.json not found at {station_rotation_json_path}. Image rotation correction will not be applied.")
    station_dict = {}
except json.JSONDecodeError:
    print(
        f"Warning: Error decoding station_rotation.json at {station_rotation_json_path}. Image rotation correction will not be applied.")
    station_dict = {}

# Regex for the new 'ori' format
# e.g., ..._ori007_5_ZHENGZHOUDONGSHANGXING.jpg
ori_format_pattern = re.compile(r'_ori(\d{3})_(\d+)_([A-Z0-9]+)$')

def parse_full_image_metadata(filepath: str) -> Dict[str, Any]:
    """
    Parse full metadata from image naming conventions:
    1. 'L' format: '..._Ld_dddd.jpg' (e.g., beijing_CR400AF-B-2118..._L7_0089.jpg)
    2. 'ori' format: '..._ori<pic-no>_<channel>_<station>.jpg' (e.g., ..._ori007_5_ZHENGZHOUDONGSHANGXING.jpg)
    3. '10X' format: '..._carSeq_10X<channel><pic-no>.jpg' (e.g., haerbin_CRH380BG-5822..._1_10X5003.jpg)

    Returns a dictionary of parsed components, or raises ValueError if parsing fails.
    """
    filename = os.path.basename(filepath)
    filename_no_ext, _ = os.path.splitext(filename)
    parts = filename_no_ext.split('_')

    # 1. Check for 'Ld_dddd' format
    if len(parts) >= 2:
        last_part_minus_one = parts[-2]
        last_part = parts[-1]

        l_digit_pattern = re.compile(r'^L\d+$')
        four_digit_pattern = re.compile(r'^\d{4}$')

        if l_digit_pattern.match(last_part_minus_one) and four_digit_pattern.match(last_part):
            return {
                'bureau': parts[0] if len(parts) > 0 else "UNKNOWN_BUREAU",
                'train_group': parts[1] if len(parts) > 1 else "UNKNOWN_TRAIN_GROUP",
                'pass_time': parts[2] if len(parts) > 2 else "UNKNOWN_PASS_TIME",
                'station': "NO_ROTATION_STATION_L",  # Explicitly set to a value that won't trigger rotation
                'row': parts[3] if len(parts) > 3 else "UNKNOWN_ROW",
                'car_seq': -1,
                'channel': 0,  # Placeholder, will not match '1'-'9' in get_rotation_flag
                'pic_no': -1
            }

    # 2. Check for new 'ori' format
    ori_match = ori_format_pattern.search(filename_no_ext)
    if ori_match:
        pic_no_str, channel_str, station_str = ori_match.groups()
        try:
            return {
                'bureau': parts[0] if len(parts) > 0 else "UNKNOWN_BUREAU", # fault-id
                'train_group': parts[1] if len(parts) > 1 else "UNKNOWN_TRAIN_GROUP", # fault-name
                'pass_time': "UNKNOWN_PASS_TIME", # Not in this format
                'station': station_str,
                'row': parts[2] if len(parts) > 2 else "UNKNOWN_ROW", # fault-only-id
                'car_seq': -1,  # Not in this format
                'channel': int(channel_str),
                'pic_no': int(pic_no_str)
            }
        except ValueError:
            raise ValueError(f"Numeric conversion failed for 'ori' format: {filename}")
        except IndexError:
             raise ValueError(f"Not enough parts for 'ori' format: {filename}")

    # 3. Standard parsing for the '10X' naming convention
    if len(parts) < 7:
        raise ValueError(f"Filename {filename} does not match 'L', 'ori', or '10X' format (expected >= 7 parts for 10X).")

    station = parts[3]
    car_seq_str = parts[-2]
    channel_pic_no_str = parts[-1]

    if not (car_seq_str.isdigit() and channel_pic_no_str.startswith('10X') and len(channel_pic_no_str) >= 7):
        raise ValueError(
            f"Last parts '{car_seq_str}_{channel_pic_no_str}' do not match expected 'A_10XBCCC' (normal) format in {filename}. This image might have an unexpected naming pattern.")

    try:
        car_seq = int(car_seq_str)
        channel = int(channel_pic_no_str[3])
        pic_no = int(channel_pic_no_str[4:])
    except ValueError:
        raise ValueError(f"Numeric conversion failed for parts: {car_seq_str}, {channel_pic_no_str} in {filename}.")

    return {
        'bureau': parts[0],
        'train_group': parts[1],
        'pass_time': parts[2],
        'station': station,
        'row': parts[4],
        'car_seq': car_seq,
        'channel': channel,
        'pic_no': pic_no
    }


# This function determines if an image needs rotation based on its channel and station.
# (This function remains unchanged, just ensure it's present in dataset_folder.py)
def get_rotation_flag(channel_str: str, station_name: str, station_map_dict: Dict) -> bool:
    # If the station name indicates an 'L' type image (which should not be rotated)
    if station_name == "NO_ROTATION_STATION_L":
        return False

    if channel_str in ('1', '2', '3', '4'):
        station_key = station_name + '_1'
    elif channel_str in ('5', '6', '7', '8', '9'):
        candidate_key = f'{station_name}_{channel_str}'
        if candidate_key in station_map_dict:
            station_key = candidate_key
        else:
            station_key = f'{station_name}_5'
    else:
        return False  # Channel not recognized or is 0 (from L-type images), no rotation

    if station_key not in station_map_dict:
        # print(f"Warning: Station key '{station_key}' not found in rotation map.") # Optional logging
        return False  # Station not in map, no rotation

    return station_map_dict[station_key] == 1  # Returns True if rotation is needed

# --- END: Custom image metadata parsing and logic ---