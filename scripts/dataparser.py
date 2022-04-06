import csv
from logging import root
from operator import gt
import cv2 as cv
import numpy as np

from noise import *
from main import ROOT_DIR


AREA_THRESHOLD = 50*50


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
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
            if area >= AREA_THRESHOLD:            
                file_data.append((filename, int(width), int(height)))

        class_list.append(file_data)
        gtFile.close()


    return class_list


def store(rootpath):
    traffic_list = readTrafficSigns(f'{rootpath}/original')
    rescount_dict = {i: len(x) for i, x in enumerate(traffic_list)}
    largest_img = {k: v for k, v in {k: v for k, v in sorted(rescount_dict.items(), key=lambda item: item[1], reverse=True)}.items()}


    masks = os.listdir(f'{rootpath}/mask')
    i = 0
    for c in list(largest_img.keys())[:10]:
        new_path = f'{rootpath}/training/{c:0>5}'
        try:
            os.mkdir(new_path)
        except:
            pass

        for img, width, height in traffic_list[c]:
            img_path = f'{rootpath}/original/{c:0>5}/{img}'
            mask_path = f'{rootpath}/mask/{masks[i]}'

            new_img = img.split('.')[0]
            cv2.imwrite(f'{new_path}/{new_img}.png', apply(img_path, mask_path, width, height))
            i = (i + 1) % len(masks)


store(f'{ROOT_DIR}/data')


