from __future__ import division
import tqdm
import torch
import torch.nn as nn
import numpy as np
import cv2


def to_cpu(tensor):
    return tensor.detach().cpu()


def letterbox_image(img, input_dim):
    """
        padding을 이용해 고정된 aspect ratio로 image를 resize
    :param img:
    :param input_dim:
    :return:
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = input_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_h, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((input_dim[1], input_dim[0], 3), 128)

    canvas[(h - new_h) // 2: (h - new_h) // 2 + new_h, (w - new_w) // 2: (w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(img, input_dim):
    """
        신경망에 넣을 이미지 준비
    :param img:
    :param input_dim:
    :return:
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (input_dim, input_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)

    return img_, orig_im, dim


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size()
    nA = pred_boxes.size(1)
    nC = pred_boxes.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)

    # Separate target values
    b, target_labels = target[:, :2].long().t()
