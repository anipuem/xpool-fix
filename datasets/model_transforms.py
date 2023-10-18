from torchvision import transforms
from PIL import Image


# torchvision.transforms主要用于常见的图形变换
def init_transform_dict(input_res=224):  # input_resolution
    tsfm_dict = {
        # test和train分别串联多个图片变换的操作
        'clip_test': transforms.Compose([
            # https://blog.csdn.net/weixin_44145782/article/details/124026928
            transforms.Resize(input_res, interpolation=Image.BICUBIC),  # 短边缩放至input_res，长宽比保持不变；interpolation选择插值方法
            transforms.CenterCrop(input_res),  # 从图片中心开始沿两边裁剪，裁剪后的图片大小为(input_res*input_res)
            # image通常在加载的时候就在三个通道归一化成[0,1]了，通过Normalize使数据符合正态分布，加速收敛
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]),
        'clip_train': transforms.Compose([
            # 随机裁剪图片,再缩放到(input_res*input_res)，scale表示最小覆盖0.5最大不变
            transforms.RandomResizedCrop(input_res, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),  # 依据概率p对PIL图片进行水平翻转，default=0.5
            # 随机改变图像的属性：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue)
            transforms.ColorJitter(brightness=0, saturation=0, hue=0),  # float数值表示偏移程度，即随机∈(1-a,1+a)，此时不变
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    }

    return tsfm_dict
