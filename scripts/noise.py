import cv2


# scales mask down to dimensions
# assumes mask is at least as large as img
# returns image (numpy array)
def apply(img_filename, mask_filenames, width, height):
    img = cv2.imread(img_filename)

    scalar = 2
    # scale mask down
    dsize = tuple([scalar * dim for dim in [width, height]])

    for mask_filename in mask_filenames:
        mask = cv2.resize(cv2.imread(mask_filename), dsize)[0:height, 0:width]
        img = cv2.add(img, mask)

    return img
