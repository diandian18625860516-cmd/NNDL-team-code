"""模型环境安装后运行；小型张量测试不需要真实数据或 GPU。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
    from salad_training.losses import BaseCompositionLoss, composition_score, extra_losses
except ImportError:
    torch = None


@unittest.skipIf(torch is None, '当前环境缺少 torch/torchvision；未验证深度学习运行')
class LossTests(unittest.TestCase):
    def test_ranking_direction_and_consistency(self):
        original = torch.tensor([.1], requires_grad=True)
        valid = torch.tensor([.4], requires_grad=True)
        invalid = torch.tensor([.2], requires_grad=True)
        consistency, ranking = extra_losses(original, valid, invalid, .1)
        self.assertAlmostEqual(consistency.item(), .09, places=6)
        self.assertAlmostEqual(ranking.item(), .3, places=6)
        ranking.backward()
        self.assertGreater(valid.grad.item(), 0)  # 梯度下降会降低合法分数。
        self.assertLess(invalid.grad.item(), 0)   # 梯度下降会提高违规分数。
        self.assertEqual(extra_losses(original, valid, torch.tensor([.8]), .1)[1].item(), 0)

    def test_raw_score_keeps_gradients_through_both_models(self):
        torch.manual_seed(42)
        models = {'comp_autoencoder': torch.nn.Conv2d(6, 6, 1),
                  'comp_unet': torch.nn.Conv2d(12, 1, 1)}
        scores = composition_score(models, torch.rand(2, 6, 8, 8))
        self.assertEqual(tuple(scores.shape), (2,))
        scores.sum().backward()
        for model in models.values():
            self.assertGreater(model.weight.grad.abs().sum().item(), 0)

    def test_base_losses_supply_gradients_without_added_objectives(self):
        reconstruction = torch.randn(2, 6, 8, 8, requires_grad=True)
        logits = torch.randn(2, 8, 8, requires_grad=True)
        target = torch.randint(0, 6, (2, 8, 8))
        mask = torch.zeros(2, 8, 8)
        mask[0, :4] = 1
        reconstruction_loss, mask_loss = BaseCompositionLoss()(reconstruction.softmax(1), target, logits, mask)
        (reconstruction_loss + mask_loss).backward()
        self.assertGreater(reconstruction.grad.abs().sum().item(), 0)
        self.assertGreater(logits.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
