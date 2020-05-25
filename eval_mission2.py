## 需要将检测得到的车牌号以xml文件的形式储存在'./Plate_dataset/AC/test/xml_pred/'，文件名一一对应
import os
import xml.etree.ElementTree as ET


if __name__ == '__main__':
    xml_gt = './data/plate_data/test/xml'
    xml_pred = './data/plate_data/test//xml_pred'
    total = 0
    pred_true = 0
    char_correct = 0
    char_total = 0
    pred_5_true = 0
    for file in os.listdir(xml_gt):
        total += 1
        anno_gt = ET.ElementTree(file=os.path.join(xml_gt, file))
        label_gt = anno_gt.find('object').find('platetext').text

        anno_pred = ET.ElementTree(file=os.path.join(xml_pred, file))
        label_pred = anno_pred.find('object').find('platetext').text
        cnt = sum([1 for i in range(6) if label_pred[i] == label_gt[i]])
        char_correct += cnt
        char_total += 6
        if cnt == 6:
            pred_true += 1
        else:
            print('Plate {} wrong, expected {}, predicted {}'.format(file.replace('.xml', ''), label_gt, label_pred))
        if cnt >= 5:
            pred_5_true += 1
            



    print('车牌预测准确率为{}'.format(pred_true/total))
    print('对5个字符以上的准确率为{}'.format(pred_5_true/total))
    print('字符识别准确率为{}'.format(char_correct/char_total))

