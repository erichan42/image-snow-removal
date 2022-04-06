import cv2
import os # don't need

# DONT NEED THIS
def apply_and_store(img_filename, mask_filename, destfilename, log_fun = print):
    """
    composites images from two files, writes result to the destfilename

    @param img_filename  
    @param log_fun  1-arg function that is invoked on a filename if cv2::imread fails while opening it
    """
    try:
        fg_img = cv2.imread(img_filename)
    except:
        log_fun(img_filename)
        return
    try:
        mask_img = cv2.imread(mask_filename)
    except:
        log_fun(mask_filename)
        return

    masked_img = cv2.add(fg_img, masked_img)
    try:
        cv2.imwrite(destfilename, masked_img)
    except:
        log_fun(destfilename)
        return

# scales mask down to dimensions
# assumes mask is at least as large as img
# returns image (numpy array)
def apply(img_filename, mask_filename, width, height):
    mask = cv2.imread(mask_filename)
    # scale mask down
    mask = mask[0:height, 0:width]
    
    img = cv2.imread(img_filename)
    masked_img = cv2.add(img, mask)

    return masked_img


# DONT NEED THIS
def apply_over_folders(mask_folder, img_folder, metadata_file, dest_folder):
    """ 
    for each file in img_folder, applies a random mask and writes to dest_folder
    """
    img_dimensions = {}
    # get dimensions from metadata file (csv) and store into dictionary
        # read all lines
    with open(metadata_file) as metadata:
        for row in metadata:
            # Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
            # split by ';'
            d = { i :[] for i in range(0, 50) }
            name, width, height, _ = row.split(';')
            # map filename to dimension tuple
            img_dimensions[name] = ()
    
    # iterate over the img_folder (class folder) directory
        # directory path
        # for file in os.listdir(img_folder):

    # pick a random mask

    # apply mask

    # write to dest folder
    