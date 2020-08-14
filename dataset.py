import numpy as np
import cv2
from random import uniform
import random

colors = [[50, 0, 0], [0, 0, 70]]


def reset_random():
    random.seed(0)


def random_xywh():
    w = uniform(0.2, 0.7)
    h = uniform(0.2, 0.7)
    x = uniform(0.1 + w / 2, 0.9 - w / 2)
    y = uniform(0.1 + h / 2, 0.9 - h / 2)
    return x, y, w, h


def fill_shape(img, x, y, w, h, color):
    rx, ry = img.shape[1], img.shape[0]
    x, y,  = int(x * rx), int(y * ry)
    ax, ay = int(w * rx / 2), int(h * ry / 2)
    cv2.ellipse(img, (x, y), (ax, ay), 0, 0, 360, color, 5)


def create_image(image_dim, n_obj):
    img = np.zeros((image_dim, image_dim, 3), np.uint8)
    target_obj = np.zeros((n_obj, 4))
    target_cls = np.zeros((len(colors), 5))
    classes = random.sample(range(len(colors)), n_obj)
    for i, c in enumerate(classes):
        x, y, w, h = random_xywh()
        fill_shape(img, x, y, w, h, colors[c])
        target_obj[i, :] = [x, y, w, h]
        target_cls[c, :] = [x, y, w, h, 1]
    return img, target_obj, target_cls


def generate_batch(batch_size, image_dim, n_obj):
    images = np.zeros((batch_size, 3, image_dim, image_dim))
    targets_obj = np.zeros((batch_size, n_obj, 4))
    targets_cls = np.zeros((batch_size, len(colors), 5))
    for i in range(batch_size):
        img, tobj, tcls = create_image(image_dim, n_obj)
        targets_obj[i, :, :] = tobj
        targets_cls[i, :, :] = tcls
        images[i, :, :, :] = img.transpose((2, 1, 0)).astype(float) / 255
    return images, targets_obj, targets_cls


def main():
    reset_random()
    image_dim = 500
    batch_size = 10
    for n in range(10):
        images, targets_obj, targets_cls = generate_batch(batch_size, image_dim, 2)
        for m, img in enumerate(images):
            img = (img.transpose((2, 1, 0)) * 255).astype(np.uint8)
            cv2.imshow("img_" + str(m), img)
            cv2.waitKey(1)
        cv2.waitKey()
        print(n, images.shape, targets_obj.shape, targets_cls.shape)


if __name__ == "__main__":
    main()
