
import gc
import multiprocessing as mp
import os
import time
from argparse import ArgumentParser
from functools import partial
from multiprocessing import cpu_count, Pool, Process
from typing import Union

import cv2
import numpy as np
import pickle
import codecs
import torch
import torch.nn.functional as F
import torchvision
import base64
import numba as nb

from .adhoc_image_dataset import AdhocImageDataset
from .classes_and_palettes import GOLIATH_CLASSES, GOLIATH_PALETTE
from PIL import Image


# from worker_pool import WorkerPool

torchvision.disable_beta_transforms_warning()

timings = {}
BATCH_SIZE = 8


def inference_model(model, imgs, dtype=torch.bfloat16):
    print( type( imgs ) )
    torch.cuda.empty_cache()
    with torch.no_grad():
        if torch.cuda.is_available():
            results = model(imgs.to(dtype).cuda())
        else:
            results = model(imgs.to(dtype))
        imgs.cpu()

    results = [r.cpu() for r in results]

    return results

# Tryon mode
@nb.njit
def replace_where(arr, needle, replace):
    arr = arr.ravel()
    needles = set(needle)
    for idx in range(arr.size):
        if arr[idx] in needles:
            arr[idx] = replace

def encode_img_to_base64(img):
    img_encode = cv2.imencode('.png', img)[1]
    img_bytes = img_encode.tobytes()
    base64_img = base64.b64encode(img_bytes)
    return base64_img

def generate_image_mask(image, result, threshold=0.3):
    image = image.data.numpy() ## bgr image

    seg_logits = F.interpolate(
        result.unsqueeze(0), size=image.shape[:2], mode="bilinear"
    ).squeeze(0)

    if seg_logits.shape[0] > 1:
        pred_sem_seg = seg_logits.argmax(dim=0, keepdim=True)
    else:
        seg_logits = seg_logits.sigmoid()
        pred_sem_seg = (seg_logits > threshold).to(seg_logits)

    pred_sem_seg = pred_sem_seg.data[0].numpy()

    mask = pred_sem_seg > 0

    white_map = np.array([2,4,5,6,7,10,11,13,14,15,16,19,20,21])
    black_map = np.array([1,3,8,9,12,17,18,22,23,24,25,26,27])

    replace_where(mask, white_map, 255)
    replace_where(mask, black_map, 0)

    img_array = mask.astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save('results/api-debug.png')
    return encode_img_to_base64(img_array)

    
def img_save_and_viz(
    image, result, output_path, classes, palette, title=None, opacity=0.5, threshold=0.3, 
):
    output_file = (
        output_path.replace(".jpg", ".png")
        .replace(".jpeg", ".png")
        .replace(".png", ".npy")
    )
    output_seg_file = (
        output_path.replace(".jpg", ".png")
        .replace(".jpeg", ".png")
        .replace(".png", "_seg.npy")
    )

    image = image.data.numpy() ## bgr image

    seg_logits = F.interpolate(
        result.unsqueeze(0), size=image.shape[:2], mode="bilinear"
    ).squeeze(0)

    if seg_logits.shape[0] > 1:
        pred_sem_seg = seg_logits.argmax(dim=0, keepdim=True)
    else:
        seg_logits = seg_logits.sigmoid()
        pred_sem_seg = (seg_logits > threshold).to(seg_logits)

    pred_sem_seg = pred_sem_seg.data[0].numpy()

    mask = pred_sem_seg > 0
    np.save(output_file, mask)
    np.save(output_seg_file, pred_sem_seg)

    num_classes = len(classes)
    sem_seg = pred_sem_seg
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    colors = [palette[label] for label in labels]

    mask = np.zeros_like(image)
    for label, color in zip(labels, colors):
        mask[sem_seg == label, :] = color
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis_image = (image_rgb * (1 - opacity) + mask * opacity).astype(np.uint8)

    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    vis_image = np.concatenate([image, vis_image], axis=1)
    cv2.imwrite(output_path, vis_image)

def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()
    

def process_image_into_dataset(img_file):
    inf_dataset = AdhocImageDataset(
        [img_file], 
        (1024,768),
        mean=[123.5, 116.5, 103.5],
        std=[58.5, 57.0, 57.5],
        )
    
    inf_dataloader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=int(os.getenv('BATCH_SIZE', 4)),
        shuffle=False
        # ,
        # num_workers=max(min(int(os.getenv('BATCH_SIZE', 4)), cpu_count()), 1) if torch.cuda.is_available() else 1,
    )

    return inf_dataset, inf_dataloader