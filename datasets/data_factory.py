from config.base_config import Config
from datasets.model_transforms import init_transform_dict
from datasets.msrvtt_dataset import MSRVTTDataset
from datasets.msvd_dataset import MSVDDataset
from datasets.lsmdc_dataset import LSMDCDataset
from torch.utils.data import DataLoader

"""
python类方法
1. 不带装饰器的实例方法，第一个参数必须是self，所以实例化类之后才可以调用
2. @staticmethod 装饰的静态方法，不需要实例化类也可以直接调用，只是职责归属的时候被抽象到类里了
"""


class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type='train'):  # config:Config类型注解，期望输入config的类型是Config，非强校验
        # 获得图像变换操作
        # 一个image对应的transformer_dict，key是'clip_train'和'clip_test'
        img_transforms = init_transform_dict(config.input_res)  # 输入分辨率，default=224
        # 获得value，即对应的一系列图像变换操作
        train_img_tfms = img_transforms['clip_train']
        test_img_tfms = img_transforms['clip_test']

        # load data
        if config.dataset_name == "MSRVTT":
            if split_type == 'train':
                # 数据集加载，并用DataLoader实现能够批量(batch_size=32)提取
                dataset = MSRVTTDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.num_workers)
            else:
                dataset = MSRVTTDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == "MSVD":
            if split_type == 'train':
                dataset = MSVDDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)
            else:
                dataset = MSVDDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=config.num_workers)
            
        elif config.dataset_name == 'LSMDC':
            if split_type == 'train':
                dataset = LSMDCDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers)
            else:
                dataset = LSMDCDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

        else:
            raise NotImplementedError
