
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


def generate_mask_color(classes_to_select):
    white_map_class_index = []
    black_map_class_index = []
    for ii, this_class in enumerate(GOLIATH_CLASSES):
        if this_class in classes_to_select:
            white_map_class_index.append(ii)
        else:
            black_map_class_index.append(ii)

    return np.array(white_map_class_index), np.array(black_map_class_index)
    

def generate_image_mask(image, result, classes, threshold=0.3):
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
    print( pred_sem_seg )

    mask = pred_sem_seg > 0
    # print( mask.shape )

    # white_map = np.array([2,4,5,6,7,10,11,13,14,15,16,19,20,21])
    # black_map = np.array([1,3,8,9,12,17,18,22,23,24,25,26,27])
    white_map, black_map = generate_mask_color(classes)

    replace_where(pred_sem_seg, white_map, 255)
    replace_where(pred_sem_seg, black_map, 0)

    img_array = pred_sem_seg.astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save('results/api-debug.png')
    return encode_img_to_base64(img_array)

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

def decode_base64_to_img(data):
    # Decode base64 string to image
    img_bytes = base64.b64decode(data.split(',')[1])
    return img_bytes
