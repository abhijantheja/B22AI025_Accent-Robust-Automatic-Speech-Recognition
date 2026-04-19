"""
Gradient Reversal Layer (GRL) for Domain-Adversarial Training.
Reference: Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Custom autograd function that reverses gradients during backpropagation."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.saved_tensors[0]
        return -alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer.
    During forward pass: identity function.
    During backward pass: multiplies gradient by -alpha.

    Alpha is annealed from 0 to 1 following the schedule from Ganin et al.:
        alpha = 2 / (1 + exp(-10 * p)) - 1
    where p is the training progress in [0, 1].
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    @staticmethod
    def get_lambda(current_step: int, total_steps: int, gamma: float = 10.0) -> float:
        """Annealed lambda schedule from Ganin et al."""
        p = current_step / max(total_steps, 1)
        return 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p)).item()) - 1.0
