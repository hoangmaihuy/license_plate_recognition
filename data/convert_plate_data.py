import os
import xml.etree.ElementTree as ET

def get_plate_data(xml_file):
    anno = ET.ElementTree(file=xml_file)
    xmin = int(anno.find('object').find('bndbox').find('xmin').text)
    ymin = int(anno.find('object').find('bndbox').find('ymin').text)
    xmax = int(anno.find('object').find('bndbox').find('xmax').text)
    ymax = int(anno.find('object').find('bndbox').find('ymax').text)
    img_height = int(anno.find('size').find('height').text)
    img_width = int(anno.find('size').find('width').text)
    #print(xmin, ymin, xmax, ymax, img_width, img_height)
    x_center, y_center = (xmax+xmin)/(2*img_width), (ymax+ymin)/(2*img_height)
    width, height = (xmax-xmin)/img_width, (ymax-ymin)/img_height
    return (x_center, y_center, width, height)

if __name__ == "__main__":
    dir_names = ['train', 'test']
    for dir_name in dir_names:
        img_dir = os.path.join('./data/plate_data', dir_name, 'images')
        label_dir = os.path.join('./data/plate_data', dir_name, 'labels')
        f1 = open('./data/plate_' + dir_name + '.txt', 'w')
        for idx in range(1, len(os.listdir(img_dir))+1):
            xml_file = os.path.join('./data/plate_data', dir_name, 'xml', str(idx)+'.xml')
            x_center, y_center, width, height = get_plate_data(xml_file)
            f2 = open(os.path.join(label_dir, str(idx) + '.txt'), 'w')
            data = f'0 {x_center} {y_center} {width} {height}'
            f2.write(data)
            f2.close()
            f1.write(os.path.join(img_dir, str(idx) + '.jpg\n'))
        f1.close()

