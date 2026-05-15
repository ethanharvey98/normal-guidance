import copy
import matplotlib.pyplot as plt
import torch
# Importing our custom module(s)
import utils

class ERMLoss(torch.nn.Module):
    def __init__(self, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, labels, **kwargs):
        
        nll = self.criterion(logits, labels)
        
        return {'loss': nll, 'nll': nll}
    
class L1Loss(torch.nn.Module):
    def __init__(self, alpha, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.criterion = criterion

    def forward(self, logits, labels, **kwargs):

        params = kwargs["params"]
        
        nll = self.criterion(logits, labels)
        penalty = (self.alpha/2) * torch.abs(params).sum()
        
        return {'loss': nll + penalty, 'nll': nll}
    
class L2Loss(torch.nn.Module):
    def __init__(self, alpha, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.criterion = criterion

    def forward(self, logits, labels, **kwargs):

        params = kwargs["params"]
        
        nll = self.criterion(logits, labels)
        penalty = (self.alpha/2) * (params**2).sum()
        
        return {'loss': nll + penalty, 'nll': nll}

class GuidedAttentionL1Loss(torch.nn.Module):
    def __init__(self, alpha, beta, criterion=torch.nn.CrossEntropyLoss(), eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion = criterion
        self.eps = eps

    def get_x(self, y):
        # y shape: [S_i, num_heads]
        return torch.arange(1, len(y) + 1, device=y.device, dtype=y.dtype).unsqueeze(1)

    def calc_mean(self, y):
        # y shape: [S_i, num_heads]
        x = self.get_x(y)
        sum_y = torch.sum(y, dim=0)
        sum_y = torch.clamp(sum_y, min=self.eps)
        mean = torch.sum(x * y, dim=0) / sum_y
        return mean

    def calc_std(self, y):
        # y shape: [S_i, num_heads]
        x = self.get_x(y)
        sum_y = torch.sum(y, dim=0)
        sum_y = torch.clamp(sum_y, min=self.eps)
        mean = torch.sum(x * y, dim=0) / sum_y
        variance = torch.sum((x ** 2) * y, dim=0) / sum_y - mean ** 2
        variance = torch.clamp(variance, min=self.eps)
        return torch.sqrt(variance)

    def forward(self, logits, labels, attn_weights, lengths, params, **kwargs):
        
        nll = self.criterion(logits, labels)
        
        with torch.no_grad():
            attn_splits = torch.split(attn_weights, lengths, dim=0)            
            js = [self.get_x(attn_weights_i) for attn_weights_i in attn_splits]
            means = [self.calc_mean(attn_weights_i) for attn_weights_i in attn_splits]
            stds = [self.calc_std(attn_weights_i) for attn_weights_i in attn_splits]
            r_hats = torch.cat([utils.normal_pdf(j, mean, std) for j, mean, std in zip(js, means, stds)], dim=0)
            rs = torch.cat([
                r_hat / torch.clamp(torch.sum(r_hat, dim=0), min=self.eps) 
                for r_hat in torch.split(r_hats, lengths, dim=0)
            ], dim=0)
            
        penalty = (self.alpha / 2) * torch.abs(params).sum()
        rs = torch.clamp(rs, min=self.eps)
        attn_weights = torch.clamp(attn_weights, min=self.eps)
        diff = (attn_weights - rs) ** 2 # Squared Euclidean distance
        #diff = rs * torch.log(rs) - rs * torch.log(attn_weights) # Forward KL
        #diff = attn_weights * torch.log(attn_weights) - attn_weights * torch.log(rs) # Reverse KL
        attn_weights_penalty = self.beta * torch.stack([
            #torch.sum(diff_i) 
            torch.mean(torch.sum(diff_i, dim=0))
            for diff_i in torch.split(diff, lengths, dim=0)
        ]).mean()
                
        return {'loss': nll + penalty + attn_weights_penalty, 'nll': nll}
        