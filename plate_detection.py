from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import numpy as np
import os
import xml.etree.ElementTree as ET
from torch.utils.tensorboard import SummaryWriter

train_img_dir = '.\\Plate_dataset\\AC\\train\\jpeg'
train_xml_dir = '.\\Plate_dataset\\AC\\train\\xml'
train_pred_dir = '.\\Plate_dataset\\AC\\train\\xml_pred'
test_img_dir = '.\\Plate_dataset\\AC\\test\\jpeg'
test_xml_dir = '.\\Plate_dataset\\AC\\test\\xml'

min_height, max_height, min_width, max_width, ratio = 500, 0, 500, 0, 0.0


def get_plate_data(xml_file):
	anno = ET.ElementTree(file=xml_file)
	label = anno.find('object').find('platetext').text
	xmin = anno.find('object').find('bndbox').find('xmin').text
	ymin = anno.find('object').find('bndbox').find('ymin').text
	xmax = anno.find('object').find('bndbox').find('xmax').text
	ymax = anno.find('object').find('bndbox').find('ymax').text
	return (label, int(xmin), int(ymin), int(xmax), int(ymax))


def write_plate_data(xml_file, platetext, xmin, ymin, xmax, ymax):
	data = f'<annotation><object>' \
	       f'<platetext>{platetext}</platetext>' \
		   f'<bndbox>'\
		   f'<xmin>{xmin}</xmin>' \
		   f'<ymin>{ymin}</ymin>' \
		   f'<xmax>{xmax}</xmax>' \
		   f'<ymax>{ymax}</ymax>' \
		   f'</bndbox>' \
	       f'</object></annotation>'
	with open(xml_file, 'w') as f:
		f.write(data)


def is_plate_like(width, height):
	eps = 0.5
	return min_width <= width <= max_width \
	       and min_height <= height <= max_height \
	       and width > height and abs(width/height - ratio) < eps


def plate_localization(img):
	ret, img_binary = cv.threshold(img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
	img_label = measure.label(img_binary)
	for region in regionprops(img_label):
		ymin, xmin, ymax, xmax = region.bbox
		width, height = xmax-xmin, ymax-ymin
		if is_plate_like(width, height):
			return (xmin, ymin, xmax, ymax)
	return (0, 0, 0, 0)

def character_segmentation():
	pass


def character_recognition():
	pass


def plate_recognition(img):
	bbox = plate_recognition(img)


def estimate_plate():
	global min_height, max_height, min_width, max_width, ratio
	num = 0
	for fn in os.listdir(train_xml_dir):
		label, xmin, ymin, xmax, ymax = get_plate_data(os.path.join(train_xml_dir, fn))
		width, height = xmax-xmin, ymax-ymin
		min_width, max_width = min(width, min_width), max(width, max_width)
		min_height, max_height = min(height, min_height), max(height, max_height)
		ratio += width/height
		num += 1
	ratio /= num
	print("Estimate plate dimension", min_width, max_width, min_height, max_height, ratio)

def train():
	writer = SummaryWriter('train')
	estimate_plate()
	idx = 0
	for fn in os.listdir(train_img_dir):
		idx += 1
		img = cv.imread(os.path.join(train_img_dir, fn), 0)
		label, xmin, ymin, xmax, ymax = get_plate_data(os.path.join(train_xml_dir, str(idx)+'.xml'))
		# gt = [xmin, ymin, xmax, ymax]
		# gt = [int(b) for b in gt]
		# label_pred = plate_recognition(img)
		xmin, ymin, xmax, ymax = plate_localization(img)
		img_patch = cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
		writer.add_image(fn, cv.cvtColor(img, cv.COLOR_BGR2RGB), global_step=0, dataformats='HWC')
		writer.add_image(fn, cv.cvtColor(img_patch, cv.COLOR_BGR2RGB), global_step=1, dataformats='HWC')
		write_plate_data(os.path.join(train_pred_dir, str(idx)+'.xml'), label, xmin, ymin, xmax, ymax)


def test():
	pass


if __name__ == '__main__':
	train()
	test()