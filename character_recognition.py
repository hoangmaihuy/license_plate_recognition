# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2 as cv

width = 20
height = 20
channel = 1
best_acc = 0

chars_dir = './data/chars_data_from_plate/'
test_dir = './data/plate_data/test/images'
output_dir = './data/plate_data/test/output'
model_path = './model'
num_to_char = os.listdir(chars_dir)


def load_data(numbers, letters, batch_size = 128, val_data_limit=50):
    print("Loading data...")
    images = np.array([]).reshape(0,height,width)
    labels = np.array([])
    train_data = val_data = []
    num = -1
    for char in os.listdir(chars_dir):
        img_dir = os.path.join(chars_dir, char)
        num += 1
        imgs = [cv.imread(os.path.join(img_dir, fn), cv.IMREAD_GRAYSCALE) for fn in os.listdir(img_dir)]
        lbs = [num] * len(imgs)
        if (num < 10 and numbers) or (num >= 10 and letters):
            images = np.append(images, imgs, axis = 0)
            if (not numbers) and letters:
                lbs = [num-10] * len(imgs)
            labels = np.append(labels, lbs, axis = 0)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)
    print("Finish loading data.")
    return (X_train, y_train), (X_test, y_test)

def train(numbers=True, letters=True):
    (train_images, train_labels), (test_images, test_labels) = load_data(numbers, letters)
    train_images = train_images.reshape((train_images.shape[0], height, width, channel))
    test_images = test_images.reshape((test_images.shape[0], height, width,channel))
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channel)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    if numbers and letters:
        model.add(layers.Dense(34, activation='softmax'))
    elif numbers:
        model.add(layers.Dense(10, activation='softmax'))
    else:
        model.add(layers.Dense(24, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test acc =', test_acc)
    if numbers and letters:
        model_name = 'all_model.h5'
    elif numbers:
        model_name = 'number_model.h5'
    else:
        model_name = 'letter_model.h5'
    model.save(os.path.join(model_path, model_name))

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


def recognize_char(img, pos):
    img = resize2Square(img, 20, cv.INTER_AREA)
    image = img.reshape((1, height, width, 1))
    if pos == 2 or pos == 3:
        preds = number_model.predict(image)[0]
        num = np.argmax(preds)
    else:
        preds = all_model.predict(image)[0]
        num = np.argmax(preds)
    return num_to_char[num]


def get_plate_text():
    for idx in range(1, 101):
        text = ''
        with open(os.path.join(output_dir, str(idx)+'_chars.txt'), 'r') as f:
            lines = f.readlines()
        if (len(lines) != 6):
            text = 'AAAAAA'
        else:
            img = cv.imread(os.path.join(output_dir, str(idx)+'_binary.jpg'), cv.IMREAD_GRAYSCALE)
            pos = 0
            for line in lines:
                xmin, ymin, xmax, ymax = map(int, line.split())
                char_img = img[ymin:ymax, xmin:xmax]
                char = recognize_char(char_img, pos)
                text = text + char
                pos += 1
        with open(os.path.join(output_dir, str(idx)+'_text.txt'), 'w') as f:
            f.write(text)
            print(f'Plate {idx} text is {text}')

if __name__ == '__main__':
    train(True, False)
    train(False, True)
    train(True, True)
    global all_model, letter_model, number_model
    all_model = tf.keras.models.load_model(os.path.join(model_path, 'all_model.h5'))
    letter_model = tf.keras.models.load_model(os.path.join(model_path, 'letter_model.h5'))
    number_model = tf.keras.models.load_model(os.path.join(model_path, 'number_model.h5'))
    get_plate_text()
