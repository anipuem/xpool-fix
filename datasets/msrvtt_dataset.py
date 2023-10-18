import os
import cv2
import sys
import torch
import random
import itertools
import numpy as np
import pandas as pd
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class MSRVTTDataset(Dataset):  # 继承Dataset类，实现自定义数据加载方式
    """
        实现需要 1. 把图片的路径和label整理到文本中 (dict)
            2. 将数据信息，解析，并存到list中 (list)
            3. 重新实现__getitem__() 函数，读取每条数据和标签，并返回
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        db_file = 'data/MSRVTT/MSRVTT_data.json'
        test_csv = 'data/MSRVTT/MSRVTT_JSFUSION_test.csv'

        # 获取video_id
        if config.msrvtt_train_file == '7k':
            train_csv = 'data/MSRVTT/MSRVTT_train.7k.csv'
        else:
            train_csv = 'data/MSRVTT/MSRVTT_train.9k.csv'

        # 标注文件
        self.db = load_json(db_file)
        if split_type == 'train':
            train_df = pd.read_csv(train_csv)
            self.train_vids = train_df['video_id'].unique()   # 返回去重之后的不同值
            self._compute_vid2caption()
            self._construct_all_train_pairs()
        else:
            self.test_df = pd.read_csv(test_csv)

    def __getitem__(self, index):  # 使实例化对象能通过instant[index]来取值
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)

        # 从视频中获得images(frames和frames从哪里sample的idxs)
        imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                         self.config.num_frames, 
                                                         self.config.video_sample_type)

        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
        }

    def __len__(self):  # 使实例化对象用len()-->返回容器中元素的个数
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.test_df)


    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        if self.split_type == 'train':
            vid, caption = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
        else:
            vid = self.test_df.iloc[index].video_id
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            caption = self.test_df.iloc[index].sentence

        return video_path, caption, vid

    
    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption in self.vid2caption[vid]:
                    self.all_train_pairs.append([vid, caption])  # 构成所有[vid, caption]s组成的的list

            
    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)  # 创建一个当key不存在时，返回默认值(list-->[])的字典
        """
        db内容为：
        {    "info":{...}
             "sentences":{
                ["caption": xxx,
                ...
                "video_id": xxx],["caption": xxx,...
            }
        }
        """
        for annotation in self.db['sentences']:  # sentences有很多list套list，所以有很多组caption和video_id
            caption = annotation['caption']
            vid = annotation['video_id']
            self.vid2caption[vid].append(caption)  # 逐个提取，构成key:vid, value:caption的字典
