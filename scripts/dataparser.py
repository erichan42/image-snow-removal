import csv
import os
import time
import cv2 as cv
import numpy as np
import shutil

from operator import gt

import scripts.noise as noise


AREA_THRESHOLD = 50*50


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath, area_threshold):
    class_list = []

    # loop over all 42 classes
    for c in range(0,43):
        file_data = []
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for filename, width, height, *rest in gtReader:
            area = int(width) * int(height)
            if area >= area_threshold:            
                file_data.append((filename, int(width), int(height)))

        class_list.append(file_data)
        gtFile.close()

    return class_list


def store_data(datapath, top, area_threshold, force, n_mask, buildpath='ml-examples', maskpath='mask'):
    if force:
        try:
            os.rmdir(f'{datapath}/{buildpath}')
        except OSError:
            i = input(f'WARNING: Directory not empty. Proceeding to delete \'{datapath}/{buildpath}\'. Continue ([y]/n)? ')
            if i.lower() in ['', 'y', 'yes']:
                shutil.rmtree(f'{datapath}/{buildpath}', ignore_errors=True)
                while os.path.isdir(f'{datapath}/{buildpath}'):
                    time.sleep(0.1)

                os.mkdir(f'{datapath}/{buildpath}')
            else:
                raise

    traffic_list = readTrafficSigns(f'{datapath}/original', area_threshold)
    rescount_dict = {i: len(x) for i, x in enumerate(traffic_list)}
    largest_img = {k: v for k, v in {k: v for k, v in sorted(rescount_dict.items(), key=lambda item: item[1], reverse=True)}.items()}

    masks = os.listdir(f'{datapath}/{maskpath}')
    i = 0
    for c in list(largest_img.keys())[:top]:
        new_path = f'{datapath}/{buildpath}/{c:0>5}'
        os.mkdir(new_path)

        for img, width, height in traffic_list[c]:
            img_path = f'{datapath}/original/{c:0>5}/{img}'
            # mask_path = f'{datapath}/maskpath/{}'
            mask_path = [f'{datapath}/{maskpath}/{mask_file}' for mask_file in _build_mask_list(masks, n_mask, i)]

            new_img = img.split('.')[0]
            cv.imwrite(f'{new_path}/{new_img}.png', noise.apply(img_path, mask_path, width, height))
            i = (i + 1) % len(masks)


def _build_mask_list(masks, n_mask, idx):
    new_mask = masks[idx:idx + n_mask]
    if idx + n_mask > len(masks):
        new_mask.extend(masks[0:idx + n_mask - len(masks)])
    return new_mask