import torch.nn as nn
import torch
import torch.nn.functional as F
from config.base_config import Config


class CLIPLoss(nn.Module):  # 用的是对比学习的InfoNCE loss
    def __init__(self):
        super().__init__()

    # https://blog.csdn.net/yyhaohaoxuexi/article/details/113824125
    # sims是内积，5对图像文本对的话，sims是5x5矩阵，对角线是配对的t-v
    def forward(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits = sims * logit_scale

        # t2v loss
        t2v_log_sm = F.log_softmax(logits, dim=1)
        # 取对角线-->获得所有正对q,k+
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        # v2t loss
        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0


class LossFactory:
    @staticmethod
    def get_loss(config: Config):
        if config.loss == 'clip':
            return CLIPLoss()
        else:
            raise NotImplemented
