import os
import ujson as json
import zipfile
import numpy as np
import pickle
import random
import torch
import shutil

# Only works with huggingface param names
def freeze_layers_clip(model, freeze_layer_num):
    assert hasattr(model, 'clip')
    assert freeze_layer_num <= 12 and freeze_layer_num >= -1

    if freeze_layer_num == -1:
        return

    for name, param in model.clip.named_parameters():
        # top layers always need to train
        if 'final_layer_norm' in name or 'text_projection' in name \
                or 'post_layernorm' in name or 'visual_projection' in name \
                or 'logit_scale' in name:
            continue # need to train
        
        elif 'text_model.encoder.layers' in name or 'vision_model.encoder.layers' in name:
            layer_num = int(name.split('.layers.')[1].split('.')[0])
            if layer_num >= freeze_layer_num:
                continue # need to train

        print(name)
        param.requires_grad = False
        

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]


def mkdirp(p):  # 如果p不存在，则创建p
    if not os.path.exists(p):
        os.makedirs(p)


def deletedir(p):  # 递归删除文件夹下的所有子文件夹和子文件，包括p本身也会删除
    if os.path.exists(p):
        shutil.rmtree(p)
