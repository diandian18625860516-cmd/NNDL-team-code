"""训练目标。保留作者基础目标的计算细节，新增目标单独开关。"""
import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class BaseCompositionLoss:
    def __init__(self):
        # 作者 get_weights 对六通道 one-hot 的数值计数，实际得到此向量。
        # 保留这一行为，避免把权重算法修正混进本次对照实验。
        self.weights = torch.tensor([1.2, 6., 1., 1., 1., 1.])
        self.dice_weight = self.weights / self.weights.sum()

    def __call__(self, reconstruction, target, logits, mask):
        # 作者先 softmax 重建输出，再交给内部含 log_softmax 的 focal loss。
        logp = F.log_softmax(reconstruction, dim=1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        focal = (-(1 - logpt.exp()).square() * logpt * self.weights.to(logp.device)[target]).mean()
        p = reconstruction.flatten(2).softmax(1)
        truth = F.one_hot(target, 6).permute(0, 3, 1, 2).flatten(2).to(p.dtype)
        coefficient = (2 * (p * truth).sum(2) + 1e-5) / (p.square().sum(2) + truth.sum(2) + 1e-5)
        # 精确保留作者 DiceLoss 的类型转换分支；GPU 上首次调用才乘权重。
        # 恢复训练时也恢复此状态，避免每次重启再次触发该分支。
        if self.dice_weight.type() != p.type():
            self.dice_weight = self.dice_weight.type_as(p)
            coefficient = coefficient * self.dice_weight * 6
        reconstruction_loss = focal + 1 - coefficient.mean()
        mask_loss = 5 * sigmoid_focal_loss(logits, mask).mean() + F.l1_loss(logits.sigmoid(), mask)
        return reconstruction_loss, mask_loss


def composition_score(models, composition):
    """可反向传播的原始组成分支分数；不是最终三分支融合测试分数。"""
    reconstruction = models['comp_autoencoder'](composition).softmax(1)
    logits = models['comp_unet'](torch.cat((composition, reconstruction), 1))
    return logits.sigmoid().flatten(1).amax(1)


def extra_losses(original_score, valid_score, invalid_score, margin):
    consistency = (valid_score - original_score).square().mean()
    ranking = F.relu(margin + valid_score - invalid_score).mean()
    return consistency, ranking
