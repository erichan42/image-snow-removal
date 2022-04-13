
import os
import cv2
import numpy as np

"""
Variables and Functions used by the various model scripts

DATA_DIR
load images,
normalize labels,
"""

# # directory of data.
# currently: this.currentdirectory.parent.data
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')


def load_corresponding_images(
    pattern_dir,
    src_dir,
    pattern_ext,
    src_ext,
    width=32, height=32,
    ):
    """
    Returns parallel lists of the subset of images in src_dir that are in pattern_dir
    (image, label)

    Scales the loaded images to widthXheight
    """
    img_list=[]
    class_name=[]
    # ignore hidden directories (starting with '.')
    for dir1 in [name for name in os.listdir(pattern_dir) if not name.startswith('.')]:
        # for every file with the mask extension
        for file in [name for name in os.listdir(os.path.join(pattern_dir, dir1)) if not name.startswith('.') and name.endswith(pattern_ext)]:
            # fix file extensions
            file = file.replace(pattern_ext, src_ext)
            # open corresponding file in src_dir
            image_path = os.path.join(src_dir, dir1,  file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            # scale to desired dimensions
            image = cv2.resize(image, (height, width),interpolation = cv2.INTER_AREA)
            # convert to numpy array of floats, normalize
            image = np.array(image)
            image = image.astype('float32')
            image /= 255 
            # add to image and label to corresponding lists
            img_list.append(image)
            class_name.append(dir1)
    return img_list, class_name


def load_images(
    img_dir,
    img_ext,
    width=32,
    height=32,
    ):
    """
    Parallel lists of images and labels from files in img_dir w/ img_ext, scaled to widthXheight
    Tuple(Images, Labels)
    """
    return load_corresponding_images(
        pattern_dir=img_dir,
        src_dir=img_dir,
        pattern_ext=img_ext,
        src_ext=img_ext,
        width=width,
        height=height,
    )


def normalize_labels(orig_vals):
    """
    takes a list of labels (numbers)
    maps the class labels s.t., for n labels, the values are between [0,n)
    returns adjusted_labels
    """
    target_dict = {k: v for v, k in enumerate(np.unique(orig_vals))}
    target_vals = [target_dict[c] for c in orig_vals]
    return target_vals


def wrap_np_array(image_list):
    """
    np.array(image_list, np.float32)
    """
    return np.array(image_list, np.float32)
