from models import BackBoneNet, RegionProposeNet, FastHead
from dataset import generate_batch, reset_random, colors
import torch
import numpy as np
import cv2


def draw_bbox(img, x, y, w, h, p, color, t):
    img = img.copy()
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    res_x, res_y = img.shape[1], img.shape[0]
    x1, y1, x2, y2 = int(res_x * x1), int(res_y * y1), int(res_x * x2), int(res_y * y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t)
    text = format(p, '.5f')
    cv2.putText(img, text, (x1, y1), 0, 1, (255, 255, 255), 1)
    return img


def detect_faster_rcnn():
    img_dim, batch_size, nr = 500, 10, 64
    
    bbn: BackBoneNet = BackBoneNet()
    rpn: RegionProposeNet = RegionProposeNet()
    fhd: FastHead = FastHead(len(colors))
    
    bbn.load_from_file("save_bbn.weights")
    rpn.load_from_file("save_rpn.weights")
    fhd.load_from_file("save_fhd.weights")
    
    reset_random()
    images, _, _ = generate_batch(batch_size, img_dim, n_obj=2)
    
    feature = bbn(torch.tensor(images).float())
    proposals = rpn.predict(feature, n_best=nr)
    preds = fhd.predict(feature, proposals)

    for img, pred in zip(images, preds.detach().numpy()):
        img = (img.transpose((2, 1, 0)) * 255).astype(np.uint8)
        ir = np.argmax(pred[:, :, 4], axis=0)
        for i in range(len(colors)):
            color = [50 + c for c in colors[i]]
            x, y, w, h, p = pred[ir[i], i, :]
            if p > 0.5:
                print(p, color)
                img = draw_bbox(img, x, y, w, h, p, color, 2)
        cv2.imshow("img", img)
        cv2.waitKey()
    print("done")


def main():
    detect_faster_rcnn()


if __name__ == "__main__":
    main()
