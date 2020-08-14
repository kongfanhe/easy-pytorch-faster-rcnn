
from dataset import generate_batch, reset_random, colors
from models import RegionProposeNet, BackBoneNet, FastHead, roi_pooling
import torch
from itertools import chain
import os
from torch.optim import Adam


def train_rpn(bbn: BackBoneNet, rpn: RegionProposeNet, bbn_active, batch_size, img_dim, its, lr):
    if bbn_active:
        optimizer = Adam(chain(bbn.parameters(), rpn.parameters()), lr)
    else:
        optimizer = Adam(rpn.parameters(), lr)
    reset_random()
    for it in range(its):
        if bbn_active:
            bbn.train()
            bbn.zero_grad()
        rpn.train()
        rpn.zero_grad()
        images, targets_obj, _ = generate_batch(batch_size, img_dim, 2)
        images = torch.tensor(images).float()
        targets_obj = torch.tensor(targets_obj).float()
        feature = bbn(images)
        loss = rpn.loss_fn(feature, targets_obj)
        loss.backward()
        optimizer.step()
        print("rpn", it, loss.item())
    return bbn, rpn


def train_fhd(bbn: BackBoneNet, rpn: RegionProposeNet, fhd: FastHead, bbn_active, batch_size, img_dim, its, lr):
    temp_file = "_temp.model"
    bbn.save_to_file(temp_file)
    bbn_backup: BackBoneNet = BackBoneNet()
    bbn_backup.load_from_file(temp_file)
    os.remove(temp_file)
    if bbn_active:
        optimizer = Adam(chain(bbn.parameters(), fhd.parameters()), lr)  # combine
    else:
        optimizer = Adam(fhd.parameters(), lr)
    reset_random()
    for it in range(its):
        fhd.train()
        fhd.zero_grad()
        if bbn_active:
            bbn.train()  # combine
            bbn.zero_grad()  # combine
        images, _, targets_cls = generate_batch(batch_size, img_dim, 2)
        images = torch.tensor(images).float()
        targets_cls = torch.tensor(targets_cls).float()
        proposals = rpn.predict(bbn_backup(images), n_best=64)
        feature = bbn(images)
        loss = fhd.loss_fn(feature, proposals, targets_cls)
        loss.backward()
        optimizer.step()
        print("fhd", it, loss.item())
    return bbn, fhd


def main():
    its, epochs, gpu = 1000, 10, 0
    
    bbn: BackBoneNet = BackBoneNet(gpu=gpu)
    rpn: RegionProposeNet = RegionProposeNet(gpu=gpu)
    fhd: FastHead = FastHead(len(colors), gpu=gpu)

    bbn.load_from_file("test_bbn.weights")
    rpn.load_from_file("test_rpn.weights")
    fhd.load_from_file("test_fhd.weights")
    
    for n in range(epochs):
        print("round", n)
        bbn, rpn = train_rpn(bbn, rpn, bbn_active=True, batch_size=5, img_dim=500, its=its, lr=1e-3)
        bbn, fhd = train_fhd(bbn, rpn, fhd, bbn_active=True, batch_size=5, img_dim=500, its=its, lr=1e-3)
        bbn, rpn = train_rpn(bbn, rpn, bbn_active=False, batch_size=5, img_dim=500, its=its, lr=1e-5)
        bbn, fhd = train_fhd(bbn, rpn, fhd, bbn_active=False, batch_size=5, img_dim=500, its=its, lr=1e-5)
        bbn.save_to_file("test_bbn.weights")
        rpn.save_to_file("test_rpn.weights")
        fhd.save_to_file("test_fhd.weights")
        
    print("done")


if __name__ == "__main__":
    main()
