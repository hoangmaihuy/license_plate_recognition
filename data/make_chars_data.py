import xml.etree.ElementTree as ET
import cv2 as cv
import numpy as np
import os

output_dir = './data/plate_data/train/output'
xml_gt = './data/plate_data/train/xml'
chars_dir = './data/chars_data_from_plate/'

def resize2Square(img, size, interpolation):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: return cv.resize(img, (size, size), interpolation)
    if h > w: dif = h
    else:     dif = w
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv.resize(mask, (size, size), interpolation)

def get_text(idx):
    anno_gt = ET.ElementTree(file=os.path.join(xml_gt, str(idx)+'.xml'))
    label_gt = anno_gt.find('object').find('platetext').text
    return label_gt

if __name__ == '__main__':
    # Apply character_segmentation to train dataset, read platetext in xml file
    # then create new characters dataset
    for idx in range(1, 582):
        img = cv.imread(os.path.join(output_dir, str(idx)+'_binary.jpg'), cv.IMREAD_GRAYSCALE)
        with open(os.path.join(output_dir, str(idx)+'_chars.txt'), 'r') as f:
            lines = f.readlines()
        text = get_text(idx)
        char_idx = 0
        if len(lines) != 6:
            continue
        for line in lines:
            xmin, ymin, xmax, ymax = map(int, line.split())
            char_img = img[ymin:ymax, xmin:xmax]
            char_img = resize2Square(char_img, 20, cv.INTER_AREA)
            char = text[char_idx]
            char_idx += 1
            char_dir = os.path.join(chars_dir, char)
            if not os.path.exists(os.path.join(char_dir)):
                os.makedirs(char_dir)
            i = len(os.listdir(char_dir)) + 1
            cv.imwrite(os.path.join(char_dir, str(i)+'.jpg'), char_img, )



