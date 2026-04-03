import math

import torch
from torch import nn


class EntropyScaleFusion(nn.Module):
    def __init__(
        self,
        num_scales=3,
        temperature=1.0,
        eps=1e-6,
        confidence_threshold=0.5,
        mlp_hidden_dim=16,
        stage=2,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.temperature = temperature
        self.eps = eps
        self.confidence_threshold = confidence_threshold
        self.stage = stage
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, num_scales),
        )

    def _uniform(self, batch_size, num_classes, device, dtype):
        return torch.full(
            (batch_size, num_classes, self.num_scales),
            1.0 / float(self.num_scales),
            device=device,
            dtype=dtype,
        )

    def forward(self, cost_volume):
        """
        Args:
            cost_volume: (B, C, T, H, W), aggregated cost volume.
        Returns:
            weights: (B, T, S), class-wise scale weights.
        """
        b, _, t, h, w = cost_volume.shape
        uniform = self._uniform(b, t, cost_volume.device, cost_volume.dtype)
        if self.stage == 1:
            return uniform

        response = cost_volume.mean(dim=1)  # B T H W
        prob = torch.softmax(response.flatten(-2) / self.temperature, dim=-1)
        entropy = -(prob * torch.log(prob + self.eps)).sum(dim=-1)  # B T
        # Clamp denominator to avoid unstable normalization on tiny maps (e.g., h*w == 1).
        entropy_denom = max(math.log(float(h * w)), 1.0)
        entropy = entropy / entropy_denom

        if t > 1:
            # Convert entropy values into rank indices, then normalize to [0, 1].
            # For small class counts (e.g., t == 2), endpoints {0, 1} are expected.
            order = torch.argsort(entropy, dim=1, descending=False)
            rank = torch.argsort(order, dim=1).float()
            rank = rank / float(t - 1)
        else:
            # Single-class case has no relative ranking.
            rank = torch.zeros_like(entropy)

        weights = self.mlp(rank.unsqueeze(-1))
        weights = torch.softmax(weights, dim=-1)

        confidence = torch.sigmoid(response.amax(dim=(-1, -2)))  # B T
        reliable = confidence >= self.confidence_threshold
        weights = torch.where(reliable.unsqueeze(-1), weights, uniform)
        return weights
