#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" this script is to divide an imaged, composed by white background and other
	multiple images, in more files"""

"""
Interactive Script to separate pictures of multiple receipts into independent images for Notilus

The script will crawl through images contained into the specified folder, run an interactive script to separate
individual receipts, and then save them in a compressed format in the specified receipt_path folder

For the interactive script, it runs in two passes:
    - First, selection of the bounding box with the mouse. When done press "q"
    - Second, itneractive segmentation. A first run shows segmented tickets in blue and background in red. If not OK,
    you can press "f" and select regions of sure foreground with the mouse, "b" for sure background, and "g" for
    rerunning the algorithm. When done, press "q"


Usage:
    python ReceiptSeparator.py <image_folder> [--receipts_path <receipt_path>]

Example:
    python interactive_grabcut.py rawpics --receipts_path receipts

"""
import argparse
import os

import cv2
import numpy as np


class DataBbox(object) :
    def __init__(self, image=None) :
        self.image = image
        self.image_copy = image.copy()
        self.ref_point = []
        self.box_exists = False
        self.cropping = True


class DataSegmentation(object) :
    def __init__(self, image=None, mask=None) :
        self.image = image
        self.image_copy = image.copy()
        self.ref_point = []
        self.current_mask = mask
        self.previous_mask = None
        self.foreground = True
        self.tinted = None


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA) :
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None :
        return image

    # check to see if the width is None
    if width is None :
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else :
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def draw_bbox(event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONDOWN :
        param.ref_point = [(x, y)]
        param.cropping = True
    elif event == cv2.EVENT_LBUTTONUP :
        param.ref_point.append((x, y))
        param.cropping = False
        param.image_copy = param.image.copy()
        cv2.rectangle(
            param.image_copy, param.ref_point[0], param.ref_point[1], (0, 255, 0), 2
        )
        cv2.imshow("image", param.image_copy)


def mark_mask(event, x, y, flags, param) :
    value = 1 if param.foreground else 0

    if event == cv2.EVENT_LBUTTONDOWN :
        param.ref_point = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP :
        param.ref_point.append((x, y))
        param.previous_mask = param.current_mask.copy()
        x1, y1 = param.ref_point[0]
        x2, y2 = param.ref_point[1]
        param.current_mask[y1 :y2, x1 :x2] = value
        param.tinted = tint_current(param.image, param.current_mask).copy()
        cv2.imshow("image", param.tinted)


def tint_current(image, mask) :
    image = image.copy() / 255.0
    tinted = tint(image, mask == 0, color=(0, 0, 1))
    tinted = tint(tinted, mask == 2, color=(0, 0.5, 1.0), alpha=0.6)
    tinted = tint(tinted, mask == 1, color=(1, 0, 0))
    tinted = tint(tinted, mask == 3, color=(1, 0.5, 0), alpha=0.6)
    return (tinted * 255).astype(np.uint8)


def tint(image, mask, color=(0, 0, 0), alpha=0.4) :
    color = np.array(color)
    image[mask] = color - (color - image[mask]) * alpha
    return image.clip(0, 1)


def get_bounding_box(image) :
    """
    Get Bounding Box to initialize GrabCut.

    Controls:
        q: quit/done.
        r: reset.
        left-click-down: start bounding box.
        left-click-up: end bounding box.
    """
    param = DataBbox(image)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_bbox, param)
    while True :
        cv2.imshow("image", param.image_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r") :
            param.image_copy = param.image.copy()
        elif key == ord("q") :
            break
    cv2.destroyAllWindows()
    rect = np.array(param.ref_point).flatten()
    if len(rect) != 4 :
        raise Exception("No Bounding Box selected")
    rect[2 :] -= rect[:2]
    return rect


def segment_image(image, bounding_box) :
    """
    Segments image using GrabCut.

    Controls:
        q: quit/done.
        r: reset.
        f: Switch to marking foreground.
        b: Switch to marking background.
        u: undo.
        g: Re-run GrabCut segmentation.
        left-click-down: start marking region.
        left-click-up: end marking region.
    """
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mask = np.zeros(image.shape[:2], np.uint8)
    print("Running GrabCut.")
    cv2.grabCut(
        image, mask, tuple(bounding_box), bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT
    )
    param = DataSegmentation(image, mask)
    param.previous_mask = mask.copy()
    param.tinted = tint_current(param.image, param.current_mask).copy()

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mark_mask, param)

    while True :
        cv2.imshow("image", param.tinted)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r") :
            param.tinted = tint_current(param.image, param.current_mask).copy()
        elif key == ord("q") :
            break
        elif key == ord("g") :
            print("Re-running grabcut.")
            cv2.grabCut(
                image,
                param.current_mask,
                tuple(bounding_box),
                bgd_model,
                fgd_model,
                5,
                cv2.GC_INIT_WITH_MASK,
            )
            param.tinted = tint_current(param.image, param.current_mask).copy()
            print("done")
        elif key == ord("f") :
            param.foreground = True
            print("Marking foreground.")
        elif key == ord("b") :
            param.foreground = False
            print("Marking background.")
        elif key == ord("u") :
            print("Undo")
            param.current_mask = param.previous_mask.copy()
            param.tinted = tint_current(param.image, param.current_mask).copy()
    cv2.destroyAllWindows()
    final_mask = np.logical_or(param.current_mask == 1, param.current_mask == 3)
    final_mask = final_mask.astype(np.uint8) * 255
    # cv2.namedWindow("Mask")
    # cv2.imshow("Mask", final_mask)
    # cv2.waitKey(3000) & 0xFF
    # cv2.destroyAllWindows()
    return final_mask


def getReceipts(mask, image, minarea=400):
    """
    Get individual receipts segmented using GrabCut
    :param mask: binarized mask showing the receipts as white areas
    :param image: source image
    :param minarea: minimal area in the image to be considered positive
    :return: a list of cropped areas of the image containing the receipts
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    cropped_receipts = []

    for cnt in contours :
        area = cv2.contourArea(cnt)

        # Shortlisting the regions based on there area.
        if area > minarea :
            xmin = cnt[:, :, 0].min()
            ymin = cnt[:, :, 1].min()
            xmax = cnt[:, :, 0].max()
            ymax = cnt[:, :, 1].max()

            cropped_receipts.append(image[ymin :ymax, xmin :xmax])

    return cropped_receipts


def load_images_from_folder(folder):
    """
    Load all images from a specified folder.
    :param folder: folder to crawl
    :return: a tuple with first a list of images, and then a list of corresponding filenames
    """
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames


def main(args) :
    image_path = args.image_folder
    receipts_path = args.receipts_path

    if receipts_path is None :
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        receipts_path = os.path.join("receipts", image_name + ".png")
    os.makedirs(os.path.join(receipts_path,""), exist_ok=True)

    print("Loading image:", image_path)
    print("Saving receipts to:", receipts_path)

    images,filenames = load_images_from_folder(image_path)

    for i in range(len(images)):
        image_orig = images[i]
        filename = os.path.splitext(os.path.basename(filenames[i]))[0]

        image = image_resize(image_orig, width=args.resize_width)

        bbox = get_bounding_box(image)
        mask = segment_image(image, bbox)

        mask_origsize = cv2.resize(mask, (image_orig.shape[1], image_orig.shape[0]))

        receipts = getReceipts(mask_origsize, image_orig, minarea=args.min_area)

        index = 0
        for receipt in receipts :
            index += 1
            receipt_path = os.path.join(receipts_path, filename + "_" + str(index) + ".jpg")
            cv2.imwrite(receipt_path, receipt, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="images", help="Folder in which the software will look for "
                                                                           "images. Default: images/")
    parser.add_argument("--receipts_path", type=str, default="receipts", help= "Folder in which the software will save"
                                                                               "individual receipts. Default: receipts/")
    parser.add_argument("--jpg_quality", type=int, default=50, help='Export quality of the JPG images of the individual '
                                                                    'receipts (range 0-100). Has an impact on the image '
                                                                    'size and quality')
    parser.add_argument("--min_area", type=int, default=400, help='Minimum area in pixels for detected blobs. Increase '
                                                                  'if you get a lot of small outputs, decrease if you'
                                                                  'miss small receipts in the output')
    parser.add_argument("--resize_width", type=int, default=500, help='Width of the resized image for the interactive '
                                                                      'process. Decrease if GrabCut is too slow or if '
                                                                      'the display doesn\'t show the whole image')
    args = parser.parse_args()
    main(args)

