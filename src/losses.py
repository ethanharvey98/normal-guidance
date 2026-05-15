import copy
from typing import Dict, Tuple
import torch
# Importing our custom module(s)
import utils

class ERMLoss(torch.nn.Module):
    def __init__(
        self, 
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ):
        super().__init__()
        self.criterion = criterion

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        nll = self.criterion(logits, labels)
        
        return {
            "loss": nll, 
            "nll": nll,
        }
    
class L1Loss(torch.nn.Module):
    def __init__(
        self, 
        alpha: float,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ):
        super().__init__()
        self.alpha = alpha
        self.criterion = criterion

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        params: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        nll = self.criterion(logits, labels)
        penalty = (self.alpha / 2) * torch.abs(params).sum()
        
        return {
            "loss": nll + penalty,
            "nll": nll,
        }
    
class L2Loss(torch.nn.Module):
    def __init__(
        self, 
        alpha: float,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    ):
        super().__init__()
        self.alpha = alpha
        self.criterion = criterion

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        params: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        nll = self.criterion(logits, labels)
        penalty = (self.alpha / 2) * (params ** 2).sum()
        
        return {
            "loss": nll + penalty,
            "nll": nll,
        }
    
class GuidedAttentionL1Loss(torch.nn.Module):
    def __init__(
        self,
        alpha: float,
        beta: float,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        eps: float = 1e-6,
        divergence: str = "squared error",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion = criterion
        self.eps = eps
        assert divergence in ["squared error", "forward kl", "reverse kl"]
        self.divergence = divergence
        
    def _get_j(
        self,
        a: torch.Tensor,
    ) -> torch.Tensor:
        # a shape: [S_i, num_heads]
        return torch.arange(1, len(a) + 1, device=a.device, dtype=a.dtype).unsqueeze(1)

    def _compute_mean(
        self,
        a: torch.Tensor,
    ) -> torch.Tensor:
        # a shape: [S_i, num_heads] -> [num_heads]
        j = self._get_j(a)
        sum_a = torch.clamp(a.sum(dim=0), min=self.eps)
        return (j * a).sum(dim=0) / sum_a

    def _compute_std(
        self,
        a: torch.Tensor,
    ) -> torch.Tensor:
        # a shape: [S_i, num_heads] -> [num_heads]
        j = self._get_j(a)
        sum_a = torch.clamp(a.sum(dim=0), min=self.eps)
        mean = (j * a).sum(dim=0) / sum_a
        variance = torch.clamp(
            (j ** 2 * a).sum(dim=0) / sum_a - mean ** 2,
            min=self.eps,
        )
        return variance.sqrt()

    def _compute_divergence(
        self,
        r: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        if self.divergence == "squared error":
            return (r - a) ** 2
        elif self.divergence == "forward kl":
            return r * (r.log() - a.log())
        elif self.divergence == "reverse kl":
            return a * (a.log() - r.log())

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attn_weights: torch.Tensor,
        lengths: Tuple[int, ...],
        params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        nll = self.criterion(logits, labels)

        with torch.no_grad():
            
            split_attn_weights = torch.split(attn_weights, lengths, dim=0)
            
            js = [self._get_j(a) for a in split_attn_weights]
            means = [self._compute_mean(a) for a in split_attn_weights]
            stds = [self._compute_std(a) for a in split_attn_weights]

            r_hats = torch.cat([
                utils.normal_pdf(j, mean, std)
                for j, mean, std in zip(js, means, stds)
            ], dim=0)
            rs = torch.cat([
                r_hat / torch.clamp(r_hat.sum(dim=0), min=self.eps)
                for r_hat in torch.split(r_hats, lengths, dim=0)
            ], dim=0)

        penalty = (self.alpha / 2) * params.abs().sum()

        rs = torch.clamp(rs, min=self.eps)
        attn_weights = torch.clamp(attn_weights, min=self.eps)

        diff = self._compute_divergence(rs, attn_weights)

        attn_penalty = self.beta * torch.stack([
            diff_i.sum(dim=0).mean()
            for diff_i in torch.split(diff, lengths, dim=0)
        ]).mean()

        return {
            "loss": nll + penalty + attn_penalty, 
            "nll": nll,
        }
    