import os
import torch
import random
import numpy as np
from config.all_config import AllConfig
from torch.utils.tensorboard.writer import SummaryWriter
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer import Trainer
from modules.optimization import AdamW, get_cosine_schedule_with_warmup


def main():
    # load命令行参数设置
    config = AllConfig()

    # pytorch transformers模型训练相关设置
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    # pytorch可以使用tensorboard进行一些可视化操作
    if not config.no_tensorboard:
        # 创建一个编写器用于保存日志，log_dir声明保存路径，已在AllConfig自动创建tb_log_dir
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None

    #  定义随机数种子，如果有随机种子，每次random的值都是固定的
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 加载Tokenizer预训练模型： 将text sequence转变为一个 id 序列
    # 使用Huggingface提供的transformer系列模型时，可以通过model.from_pretrained函数来加载预训练模型
    if config.huggingface:
        from transformers import CLIPTokenizer
        # 加载"openai/clip-vit-base-patch32"对应的CLIPTokenizer；TOKENIZERS_PARALLELISM为transformer需要的设置
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    else:
        from modules.tokenization_clip import SimpleTokenizer
        tokenizer = SimpleTokenizer()

    # data load-->返回DataLoader
    train_data_loader = DataFactory.get_data_loader(config, split_type='train')
    valid_data_loader = DataFactory.get_data_loader(config, split_type='test')
    # model load-->返回CLIPBaseline/CLIPTransformer
    model = ModelFactory.get_model(config)

    # 获得储存"R1", "R5", "MedR"等信息的dict
    if config.metric == 't2v':  # default是't2v'
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented

    # 获取model所有的参数
    params_optimizer = list(model.named_parameters())
    # 根据参数名字里有"clip"/"clip."做grouped
    clip_params = [p for n, p in params_optimizer if "clip." in n]
    noclip_params = [p for n, p in params_optimizer if "clip." not in n]
    
    optimizer_grouped_params = [  # 把不同的训练参数分别赋予不同的学习率
        {'params': clip_params, 'lr': config.clip_lr},
        {'params': noclip_params, 'lr': config.noclip_lr}
    ]
    # weight_decay：L2正则化系数，为了使模型参数趋近于0防止过拟合，config.weight_decay-->越大，限制趋近于0效果越强
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)

    # 总训练步数=epoch*每个epoch分的批次
    num_training_steps = len(train_data_loader) * config.num_epochs
    # from ResNet的学习率预热方法，在训练开始的时候先选择使用一个较小的学习率，训练了一些steps，再修改为预先设置的学习来进行训练
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    # loss使用CLIPLoss
    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=scheduler,
                      writer=writer,
                      tokenizer=tokenizer)

    # 调用base_trainer的train()方法
    trainer.train()

    if not config.no_tensorboard:
        # 用完writer后关闭
        writer.close()


if __name__ == '__main__':
    main()
