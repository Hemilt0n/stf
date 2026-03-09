from torch import nn
import torch


class TargetReferencePreference(nn.Module):
    """Measure whether prediction is closer to target (fine_img_02) or reference (fine_img_01).

    TRP = (d(pred, ref) - d(pred, tgt)) / (d(pred, ref) + d(pred, tgt) + eps)
    Positive value means prediction is closer to target than reference.
    """

    __name__ = 'TRP'

    def __init__(
        self,
        distance: str = 'l1',
        eps: float = 1e-6,
        change_aware: bool = False,
        change_power: float = 1.0,
    ):
        super().__init__()
        if distance not in {'l1', 'l2'}:
            raise ValueError(f'Unsupported distance: {distance}')
        self.distance = distance
        self.eps = eps
        self.change_aware = change_aware
        self.change_power = change_power

    def _distance(self, a, b, weight=None):
        if self.distance == 'l1':
            diff = torch.abs(a - b)
        else:
            diff = torch.square(a - b)

        if weight is None:
            return diff.mean()

        weighted = diff * weight
        return weighted.sum() / weight.sum().clamp(min=self.eps)

    def forward(self, gt, pred, reference):
        weight = None
        if self.change_aware:
            # emphasize areas with larger temporal change
            change_map = torch.mean(torch.abs(gt - reference), dim=1, keepdim=True)
            norm = change_map.mean(dim=(-2, -1), keepdim=True).clamp(min=self.eps)
            weight = (change_map / norm).pow(self.change_power)

        d_ref = self._distance(pred, reference, weight=weight)
        d_tgt = self._distance(pred, gt, weight=weight)
        return (d_ref - d_tgt) / (d_ref + d_tgt + self.eps)


TRP = TargetReferencePreference
