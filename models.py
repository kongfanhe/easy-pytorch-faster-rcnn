from torch import nn
import torch
import numpy as np
import torch.nn.functional as fn
import os

feat_depth, roi_dim, fc_neuron = 256, 3, 1024


def get_device(gpu):
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device("cuda:" + str(gpu))
    return torch.device("cpu")


def bbox_iou(box_a, box_b):
    xa, ya, wa, ha = box_a[..., 0], box_a[..., 1], box_a[..., 2], box_a[..., 3]
    xb, yb, wb, hb = box_b[..., 0], box_b[..., 1], box_b[..., 2], box_b[..., 3]
    x1, y1, x2, y2 = xa - wa / 2, ya - ha / 2, xa + wa / 2, ya + ha / 2
    x3, y3, x4, y4 = xb - wb / 2, yb - hb / 2, xb + wb / 2, yb + hb / 2
    x5 = torch.max(x1, x3)
    y5 = torch.max(y1, y3)
    x6 = torch.min(x2, x4)
    y6 = torch.min(y2, y4)
    w = torch.clamp(x6 - x5, min=0)
    h = torch.clamp(y6 - y5, min=0)
    inter_area = w * h
    b1_area = (x2 - x1) * (y2 - y1)
    b2_area = (x4 - x3) * (y4 - y3)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def anchor_bbox(anchor, n_grid):
    # anchor_tensor: [na, ng, ng, 4(x/y/w/h)]

    na, ng = len(anchor), n_grid
    grids = np.linspace(0.5 / ng, (ng - 0.5) / ng, ng)
    gy, gx = np.meshgrid(grids, grids)
    x = torch.tensor(gx).view(1, ng, ng).repeat(na, 1, 1).float()
    y = torch.tensor(gy).view(1, ng, ng).repeat(na, 1, 1).float()
    a = torch.tensor(anchor).view(na, 1, 1, 2).repeat(1, ng, ng, 1)
    w = a[:, :, :, 0]
    h = a[:, :, :, 1]
    anchor_tensor = torch.stack((x, y, w, h), dim=-1)
    return anchor_tensor


def valid_bbox(bbox):
    x, y, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    x1, x2, y1, y2 = x - w / 2, x + w / 2, y - h / 2, y + h / 2
    valid_x = torch.bitwise_and(torch.bitwise_and(0 <= x1, x1 <= x2), x2 <= 1)
    valid_y = torch.bitwise_and(torch.bitwise_and(0 <= y1, y1 <= y2), y2 <= 1)
    valid = torch.bitwise_and(valid_x, valid_y)
    return valid


def roi_pooling(feature, proposals):
    # feature: [nb, fd, ng, ng]
    # proposals: [nb, nr, 5(x/y/w/h/obj)]
    # rois: [nb, nr, fd * rd * rd]

    nb, nr, ng = proposals.size(0), proposals.size(1), feature.size(2)
    fd, rd = feat_depth, roi_dim
    p = proposals[..., 0:4]
    valid = valid_bbox(p)
    x, y, w, h = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    p = (torch.stack((x - w / 2, x + w / 2, y - h / 2, y + h / 2), dim=-1) * ng).long()
    rois = torch.zeros((nb, nr, fd, rd, rd)).float()
    for i in range(nb):
        for j in range(nr):
            if valid[i, j]:
                _x1, _x2, _y1, _y2 = p[i, j, 0], p[i, j, 1], p[i, j, 2], p[i, j, 3]
                im = feature[i, :, _x1:(_x2 + 1), _y1:(_y2 + 1)]
                rois[i, j, :, :, :] = fn.adaptive_max_pool2d(im, (rd, rd))
    rois = rois.view(nb, nr, -1)
    return rois

        
def output_to_real(outputs, proposals):
    p = proposals
    t = outputs
    xp, yp, wp, hp = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    tx, ty, tw, th = t[..., 0], t[..., 1], t[..., 2], t[..., 3]
    x = tx * wp + xp
    y = ty * hp + yp
    w = torch.exp(tw) * wp
    h = torch.exp(th) * hp
    obj = t[..., 4] + 0 * x
    real = torch.stack((x, y, w, h, obj), dim=-1)
    return real


def real_to_output(real, proposals):
    p = proposals
    r = real
    xp, yp, wp, hp = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    x, y, w, h = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    tx = (x - xp) / (wp + 1e-30)
    ty = (y - yp) / (hp + 1e-30)
    tw = torch.log((w + 1e-30) / (wp + 1e-30))
    th = torch.log((h + 1e-30) / (hp + 1e-30))
    obj = r[..., 4] + 0 * tx
    outputs = torch.stack((tx, ty, tw, th, obj), dim=-1)
    return outputs
    
    
class Net(nn.Module):

    def __init__(self, gpu):
        super().__init__()
        self.device = get_device(gpu)

    def save_to_file(self, file):
        state = self.state_dict()
        torch.save(state, file)

    def load_from_file(self, file):
        if os.path.exists(file):
            state = torch.load(file, map_location=self.device)
            self.load_state_dict(state)
        else:
            print("cannot load net file: " + file)


class BackBoneNet(Net):

    def __init__(self, gpu=-1):
        super().__init__(gpu)
        layers = list()
        layers.append(nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))
        layers.append(nn.Conv2d(96, 256, kernel_size=5, stride=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))
        layers.append(nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(384, feat_depth, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.cnn = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, images):
        images = images.to(self.device)
        feature = self.cnn(images)
        return feature


class RegionProposeNet(Net):
    anchor = [[0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [0.8, 0.8]]

    def __init__(self, gpu=-1):
        super().__init__(gpu)
        layers = list()
        layers.append(nn.Conv2d(feat_depth, feat_depth, 3, 1, 1))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(feat_depth, 5 * len(self.anchor), 1, 1, 0))
        self.cnn = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, feature):
        # feature: [nb, feat_depth, ng, ng]
        # proposals: [nb, na, ng, ng, 5(x/y/w/h/obj)]

        nb, ng, na = feature.size(0), feature.size(2), len(self.anchor)
        p = self.cnn(feature).view(nb, 5, na, ng, ng).permute((0, 2, 3, 4, 1))
        a = anchor_bbox(self.anchor, ng).to(self.device)
        ax, ay, aw, ah = a[None, ..., 0], a[None, ..., 1], a[None, ..., 2], a[None, ..., 3]
        x = p[..., 0] * aw + ax
        y = p[..., 1] * ah + ay
        w = torch.exp(p[..., 2]) * aw
        h = torch.exp(p[..., 3]) * ah
        obj = torch.sigmoid(p[..., 4])
        proposals = torch.stack((x, y, w, h, obj), dim=-1)
        return proposals

    def predict(self, feature, n_best=None):
        # feature: [nb, feat_depth, ng, ng]
        # proposals: [nb, nr, 5(x/y/w/h/obj)]
        
        nb, ng, na = feature.size(0), feature.size(2), len(self.anchor)
        nr = na * ng * ng
        feature = feature.to(self.device)
        proposals = self(feature).view(nb, nr, 5)
        idx = torch.argsort(- proposals[:, :, 4], dim=1)
        proposals = torch.gather(proposals, 1, idx[:, :, None].repeat(1, 1, 5))
        if n_best is not None:
            proposals = proposals[:, :n_best, :].contiguous()
        return proposals

    def loss_fn(self, feature, targets):
        # feature: [nb, feat_depth, ng, ng]
        # targets: [nb, no, 4(x/y/w/h)]

        proposals = self(feature)
        gt = targets.to(self.device)
        p = proposals.to(self.device)
        nb, na, ng, no = p.size(0), p.size(1), p.size(2), gt.size(1)
        anchors = anchor_bbox(self.anchor, ng)[None, :, :, :, :].repeat(nb, 1, 1, 1, 1).to(self.device)
        anchors_obj = anchors.view(nb, na, ng, ng, 1, 4).repeat(1, 1, 1, 1, no, 1)
        gt_roi = gt.view(nb, 1, 1, 1, no, 4).repeat(1, na, ng, ng, 1, 1)
        iou_obj = bbox_iou(anchors_obj, gt_roi)
        iou_idx = torch.argmax(iou_obj, dim=-1, keepdim=True)
        iou_idx_t = iou_idx.view(nb, na, ng, ng, 1, 1).repeat(1, 1, 1, 1, 1, 4)
        gt_active = torch.gather(gt_roi, -2, iou_idx_t).view(nb, na, ng, ng, 4)
        iou_active = torch.gather(iou_obj, -1, iou_idx).view(nb, na, ng, ng)
        valid = valid_bbox(anchors)
        mask_obj = torch.bitwise_and(iou_active > 0.5, valid)
        mask_no_obj = torch.bitwise_and(torch.bitwise_and(iou_active > 0, iou_active < 0.1), valid)
        loss_reg = torch.sum((p[..., 0:4] - gt_active) ** 2, dim=-1)
        loss_obj = (p[..., 4] - 1) ** 2
        loss_no_obj = (p[..., 4] - 0) ** 2
        loss_1 = torch.sum(loss_reg * mask_obj)
        loss_2 = torch.sum(loss_obj * mask_obj)
        loss_3 = torch.sum(loss_no_obj * mask_no_obj)
        loss = loss_1 + loss_2 + loss_3
        return loss


class FastHead(Net):

    def __init__(self, n_cls, gpu=-1):
        super().__init__(gpu)
        layers = list()
        layers.append(nn.Linear(feat_depth * roi_dim * roi_dim, fc_neuron))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(fc_neuron, fc_neuron))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(fc_neuron, n_cls * 5))
        self.fc = nn.Sequential(*layers)
        self.n_cls = n_cls
        self.to(self.device)

    def forward(self, rois):
        # rois: [nb, nr, fd*rd*rd]
        # outputs: [nb, nr, nc, 5(tx/ty/tw/th/obj)]

        nb, nr, nc = rois.size(0), rois.size(1), self.n_cls
        rois = rois.to(self.device)
        outputs = self.fc(rois)
        outputs = outputs.view(nb, nr, nc, 5)
        outputs[..., 4] = torch.sigmoid(outputs[..., 4])
        return outputs

    def predict(self, feature, proposals):
        # feature: [nb, fd, ng, ng]
        # proposals: [nb, nr, 5(x/y/w/h/obj)]
        # preds: [nb, nr, nc, 5(x/y/w/h/obj)]

        nb, nr, nc = proposals.size(0), proposals.size(1), self.n_cls
        rois = roi_pooling(feature, proposals)
        outputs = self(rois).view(nb, nr, nc, 5)
        preds = output_to_real(outputs, proposals[:, :, None, :])
        return preds
    
    def loss_fn(self, feature, proposals, targets):
        # feature: [nb, fd, ng, ng]
        # proposals: [nb, nr, 5(x/y/w/h/obj)]
        # targets: [nb, nc, 5(x/y/w/h/obj)]

        nb, nr, nc = proposals.size(0), proposals.size(1), self.n_cls

        feature = feature.to(self.device)
        props = proposals.to(self.device)
        rois = roi_pooling(feature, proposals)
        targets = targets.to(self.device)
        outpus = self(rois).view(nb, nr, nc, 5)
        
        tgets = real_to_output(targets[:, None, :, :], props[:, :, None, :])  # [nb, nr, nc, 5]
        
        valid = valid_bbox(props)  # [nb, nr]
        ious = bbox_iou(props[:, :, None, :], targets[:, None, :, :])  # [nb, nr, nc]

        loss = torch.sum(valid[:, :, None] * (targets[:, None, :, 4] == 1) * 
                ((ious > 0.7) * torch.sum((outpus - tgets) ** 2, dim=-1) + 
                (ious < 0.3) * (outpus[..., 4] - 0)**2))

        return loss


def main():
    nb, n_cls, img_dim, n_obj = 10, 3, 500, 2

    images = torch.zeros((nb, 3, img_dim, img_dim)).float()
    targets_obj = torch.zeros((nb, n_obj, 4)).float()
    targets_cls = torch.zeros((nb, n_cls, 5)).float()

    bbn: BackBoneNet = BackBoneNet()
    rpn: RegionProposeNet = RegionProposeNet()
    fhd: FastHead = FastHead(n_cls)

    feature = bbn(images)
    loss_rpn = rpn.loss_fn(feature, targets_obj)
    print(images.shape, feature.shape, loss_rpn)

    proposals = rpn.predict(feature, n_best=64)
    loss_fhd = fhd.loss_fn(feature, proposals, targets_cls)
    print(proposals.shape, loss_fhd)

    print("done")


if __name__ == "__main__":
    main()
