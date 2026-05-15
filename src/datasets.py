from typing import Callable, Tuple
import torch
# Importing our custom module(s)
import utils

class MILPathDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        path: list, 
        y: torch.Tensor,
        transform: Callable,
    ):
        super().__init__()
        self.path = path
        self.y = y
        self.transform = transform

    def __len__(
        self,
    ) -> int:
        return len(self.path)
    
    def __getitem__(
        self, 
        index: int,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        x = self.transform(self.path[index])
        return x, len(x), self.y[index]
    
class MILTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        x: torch.Tensor, 
        lengths: Tuple, 
        y: torch.Tensor,
        #lengths_y: torch.Tensor,
    ):
        super().__init__()
        self.x = x
        self.x_split = torch.split(x, lengths)
        self.lengths = lengths
        self.y = y
        #self.lengths_y = lengths_y

    def __len__(
        self,
    ) -> int:
        return len(self.lengths)
    
    def __getitem__(
        self, 
        index: int,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        return self.x_split[index], self.lengths[index], self.y[index]
        #return self.x_split[index], self.lengths[index], self.y[index], self.lengths_y[index]

class ShiftedMeanMILDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        n: int, 
        r: int = 12, 
        s_low: int = 20, 
        s_high: int = 60, 
        k: int = 1, 
        m: int = 768, 
        p_y1: float = 0.5, 
        delta: float = 1.0, 
        mu: float = 0.0, 
        sigma: float = 1.0, 
        seed: int = 42,
    ):
        super().__init__()
        
        self.n = n
        self.r = r
        self.s_low = s_low
        self.s_high = s_high
        self.k = k
        self.m = m
        self.p_y1 = p_y1
        self.delta = delta
        self.mu = mu
        self.sigma = sigma
        self.seed = seed
        
        self.generator = torch.Generator().manual_seed(self.seed)
        self.lengths = tuple(torch.randint(self.s_low, self.s_high + 1, (self.n,), generator=self.generator).tolist())
        self.h = self.mu + self.sigma * torch.randn(sum(self.lengths), self.m, generator=self.generator)
        self.u = torch.cat([torch.randint(0, s_i - self.r + 1, (1,), generator=self.generator) for s_i in self.lengths])
        self.y = torch.bernoulli(self.p_y1 * torch.ones(size=(self.n,)), generator=self.generator).int().reshape(-1, 1).float()
        self.h_split = torch.split(self.h, self.lengths)
        
        for i, h_i in enumerate(self.h_split):
            if self.y[i] == 1:
                h_i[self.u[i]:self.u[i] + self.r, 0:self.k] += self.delta
                
    def __len__(
        self,
    ) -> int:
        return len(self.lengths)

    def __getitem__(
        self, 
        index: int,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        return self.h_split[index], self.lengths[index], self.y[index]

    def p_h_given_u_y1(
        self,
        index: int,
    ) -> torch.Tensor:
        h_ik = self.h_split[index][:, 0:self.k]
        s_i = self.lengths[index]
        p_h_given_u_y1 = torch.stack([torch.stack([torch.stack([utils.normal_pdf(h_ik[j, k], self.mu + self.delta) if j >= u and j < (u + self.r) else utils.normal_pdf(h_ik[j, k]) for k in range(self.k)]) for j in range(s_i)]) for u in range(s_i - self.r + 1)])
        return torch.prod(torch.prod(p_h_given_u_y1, dim=-1), dim=-1)
    
    def p_y1_given_h(
        self, 
        index: int,
    ) -> torch.Tensor:
        h_ik = self.h_split[index][:, 0:self.k]
        s_i = self.lengths[index]
        p_h_given_y0 = torch.prod(torch.stack([utils.normal_pdf(h_ik[j]) for j in range(s_i)])) * (1.0 - self.p_y1)
        p_u = (1 / (s_i - self.r + 1)) * torch.ones(size=(s_i - self.r + 1,))
        p_h_given_y1 = torch.sum(self.p_h_given_u_y1(index) * p_u, dim=-1) * self.p_y1
        p_y1_given_h = p_h_given_y1 / (p_h_given_y0 + p_h_given_y1)
        return p_y1_given_h

    def p_y_j1_given_h(
        self,
        index: int,
    ) -> torch.Tensor:
        s_i = self.lengths[index]
        p_h_given_u_y1 = self.p_h_given_u_y1(index)
        p_u_given_h_y1 = p_h_given_u_y1 / p_h_given_u_y1.sum()
        scores = torch.zeros(s_i)
        for u in range(s_i - self.r + 1):
            scores[u:u + self.r] += p_u_given_h_y1[u]
        return scores    
    