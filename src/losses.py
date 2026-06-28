from __future__ import annotations


class BCEDiceLoss:
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, eps: float = 1e-6) -> None:
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps
        try:
            import torch  # type: ignore

            self._torch = torch
            self._bce = torch.nn.BCEWithLogitsLoss()
        except Exception:
            self._torch = None
            self._bce = None

    def __call__(self, logits, targets):
        if self._torch is None:
            raise ImportError("BCEDiceLoss requires torch")
        probs = self._torch.sigmoid(logits)
        bce = self._bce(logits, targets)
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + self.eps) / (denominator + self.eps)).mean()
        return self.bce_weight * bce + self.dice_weight * dice
