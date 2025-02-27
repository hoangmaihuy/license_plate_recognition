import cv2 as cv
import numpy as np
import os
import xml.etree.ElementTree as ET

img_dir = './data/plate_data/test/images'
output_dir = './data/plate_data/test/output'
xml_gt = './data/plate_data/test/xml'

def get_bbox(idx):
    # Use predict bbox
    '''
    with open(os.path.join(output_dir, str(idx)+'.txt'), 'r') as f:
        xmin, ymin, xmax, ymax = map(int, f.readline().split())
    '''
    # Use ground truth bbox
    anno_gt = ET.ElementTree(file=os.path.join(xml_gt, str(idx)+'.xml'))
    xmin = int(anno_gt.find('object').find('bndbox').find('xmin').text)
    ymin = int(anno_gt.find('object').find('bndbox').find('ymin').text)
    xmax = int(anno_gt.find('object').find('bndbox').find('xmax').text)
    ymax = int(anno_gt.find('object').find('bndbox').find('ymax').text)
    return xmin, ymin, xmax, ymax

# Remove noise pixels at four borders
def remove_border(img):
    top, bot, left, right = 6, 4, 3, 3
    img[:top,:] = 0
    img[:,:left] = 0
    img[-bot:, :] = 0
    img[:, -right:] = 0

if __name__ == '__main__':
    correct = 0
    for idx in range(1, len(os.listdir(img_dir))+1):
        img = cv.imread(os.path.join(img_dir, str(idx) + '.jpg'))
        xmin, ymin, xmax, ymax = get_bbox(idx)
        img = img[ymin:ymax, xmin:xmax]
        img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # License plate binary image
        _, thresh_inv = cv.threshold(img_grey, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        remove_border(thresh_inv)
        ctrs, _ = cv.findContours(thresh_inv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Sort contours by top-left bounding box corner
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv.boundingRect(ctr)[0]) 
        # License plate area
        img_area = img.shape[0]*img.shape[1]
        # License plate characters positions, array of (xmin, ymin, w, h) 
        chars = []
        for ctr in sorted_ctrs:
            x, y, w, h = cv.boundingRect(ctr)
            roi_area = w*h
            roi_ratio = roi_area/img_area
            box_ratio = w/h
            if box_ratio < 0.1 or h < 10:
                continue
            # Divide large bounding box into smaller ones based on box_ratio
            if roi_ratio >= 0.015:
                div = 1
                if box_ratio <= 0.83: 
                    div = 1
                elif box_ratio <= 1.4: 
                    div = 2
                elif box_ratio <= 1.9: 
                    div = 3
                elif box_ratio <= 2.6: 
                    div = 4
                elif box_ratio <= 3.2:
                    div = 5
                else:
                    div = 6
                w = int(w / div)
                for i in range(div):
                    chars.append((x+i*w, y, w, h))
        if len(chars) == 7:
            x, y, w, h = chars[-1]
            box_ratio = w/h
            if box_ratio < 0.3:
                chars.pop(-1) # Remove noise in the right
            else:
                chars.pop(0) # Remove noise in the left
        # Save result into output_dir/idx_chars.txt
        f = open(os.path.join(output_dir, str(idx)+'_chars.txt'), 'w')
        for char in chars:
            x, y, w, h = char
            f.write(f'{x} {y} {x+w} {y+h}\n')
            img = cv.rectangle(img, (x, y), (x+w, y+h), (90, 0, 255), 1)
        f.close()
        if len(chars) == 6:
            correct += 1
        else:
            print('Plate {} segmentation wrong, {} chars'.format(idx, len(chars)))
        cv.imwrite(os.path.join(output_dir, str(idx)+'_masked.jpg'), img)
        cv.imwrite(os.path.join(output_dir, str(idx)+'_grey.jpg'), img_grey)
        cv.imwrite(os.path.join(output_dir, str(idx)+'_binary.jpg'), thresh_inv)
    print("Correct =", correct)
