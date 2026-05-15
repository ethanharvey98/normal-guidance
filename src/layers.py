from typing import Tuple, Optional
import torch
# Importing our custom module(s)
import utils

class Max(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = torch.cat([
            torch.max(x_i, dim=0, keepdim=True).values 
            for x_i in torch.split(x, lengths)
        ])
        return out, None
    
class Mean(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = torch.cat([
            torch.mean(x_i, dim=0, keepdim=True) 
            for x_i in torch.split(x, lengths)
        ])
        attn_weights = torch.cat([
            torch.ones(size=(length, 1), device=x.device) / length 
            for length in lengths
        ])
        return out, attn_weights
    
class ABMIL(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_dim, out_features=1),
        )

    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_logits = self.mlp(x)
        attn_weights = torch.cat([
            torch.nn.functional.softmax(attn_logits_i, dim=0) 
            for attn_logits_i in torch.split(attn_logits, lengths)
        ])
        out = attn_weights * x
        out = torch.cat([
            torch.sum(out_i, dim=0, keepdim=True) 
            for out_i in torch.split(out, lengths)
        ])
        return out, attn_weights
        
class InstanceConv1d(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        kernel_size: int = 3
    ):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, groups=in_features, padding="same")
        
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Tuple[int, ...],
    ) -> torch.Tensor:
        out = torch.cat([
            self.conv(x_i.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
            for x_i in torch.split(x, lengths)
        ])
        return out
    
class PPEG(torch.nn.Module):
    def __init__(
        self, 
        in_features: int,
    ):
        super().__init__()
        self.proj1 = InstanceConv1d(in_features=in_features, kernel_size=3)
        self.proj2 = InstanceConv1d(in_features=in_features, kernel_size=5)
        self.proj3 = InstanceConv1d(in_features=in_features, kernel_size=7)

    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Tuple[int, ...],
    ) -> torch.Tensor:
        # Split class tokens and features
        cls_token = torch.stack([
            x_i[0,:]
            for x_i in torch.split(x, lengths)
        ])
        x = torch.cat([
            x_i[1:,:] 
            for x_i in torch.split(x, lengths)
        ])
        lengths = tuple(length - 1 for length in lengths)
        # Fuse together different spatial information
        out = x + self.proj1(x, lengths) + self.proj2(x, lengths) + self.proj3(x, lengths)
        out = torch.cat([
            torch.cat((cls_token[i].unsqueeze(0), out_i))
            for i, out_i in enumerate(torch.split(out, lengths))
        ])
        return out
    
class TransformerLayer(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        num_heads: int = 8
    ):
        super().__init__()
        self.norm = torch.nn.LayerNorm(normalized_shape=in_features)
        self.attn = torch.nn.MultiheadAttention(embed_dim=in_features, num_heads=num_heads)
        
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        norm_x = self.norm(x)
        out, attn_weights = zip(*[
            self.attn(x_i, x_i, x_i, need_weights=True, average_attn_weights=False) 
            for x_i in torch.split(norm_x, lengths)
        ])
        out = x + torch.cat(out)
        return out, attn_weights
    
class TransMIL(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        num_heads: int = 8
    ):
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.randn(size=(1, in_features,)))
        self.layer1 = TransformerLayer(in_features=in_features, num_heads=num_heads)
        self.pos_layer = PPEG(in_features=in_features)        
        self.layer2 = TransformerLayer(in_features=in_features, num_heads=num_heads)
        #self.norm = torch.nn.LayerNorm(normalized_shape=in_features)

    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Concatenate class token
        x = torch.cat([
            torch.cat((self.cls_token, x_i)) 
            for x_i in torch.split(x, lengths)
        ])
        lengths = tuple(length + 1 for length in lengths)
        # First transformer layer
        out, _ = self.layer1(x, lengths)
        # Pyramid position encoding generator layer
        out = self.pos_layer(out, lengths)
        # Second transformer layer
        out, attn_weights = self.layer2(out, lengths)
        # Get class token
        #out = torch.stack([
        #    self.norm(out_i[0,:]) 
        #    for out_i in torch.split(out, lengths)
        #])
        out = torch.stack([
            out_i[0,:] 
            for out_i in torch.split(out, lengths)
        ])
        # Get attention weights from class token
        # Remove attention weight for class token
        attn_weights = torch.cat([
            attn_weights_i[:,0,1:].T
            for attn_weights_i in attn_weights
        ])
        return out, attn_weights
        #return out, torch.mean(attn_weights, dim=1, keepdim=True)
    
class Sm(torch.nn.Module):
    def __init__(
        self, 
        alpha: float = 0.5, 
        learnable_alpha: bool = True, 
        neighbors: int = 1, 
        self_loop: bool = False
    ):
        super().__init__()
        self.neighbors = neighbors
        self.self_loop = self_loop
        if learnable_alpha:
            self.raw_alpha = torch.nn.Parameter(utils.inv_sigmoid(torch.tensor(alpha, dtype=torch.float32)))
        else:
            self.register_buffer("raw_alpha", utils.inv_sigmoid(torch.tensor(alpha, dtype=torch.float32)))

    @property
    def alpha(
        self,
    ) -> torch.Tensor:
        return torch.nn.functional.sigmoid(self.raw_alpha)

class ApproxSm(Sm):
    def __init__(
        self,
        alpha: float = 0.5,
        learnable_alpha: bool = True,
        neighbors: int = 1,
        num_steps: int = 10,
        self_loop: bool = False,
        use_matrix: bool = True,
    ):
        super().__init__(
            alpha = alpha, 
            learnable_alpha = learnable_alpha, 
            neighbors = neighbors, 
            self_loop = self_loop, 
        )
        self.num_steps = num_steps
        self.use_matrix = use_matrix
        self._cached_A = None
        self._cached_length = None

    def _get_adjacency_matrix(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Cache the adjacency matrix if length hasn't changed
        if self._cached_A is not None and self._cached_length == length:
            return self._cached_A.to(device=device, dtype=dtype)

        # Build normalized adjacency matrix
        A = torch.zeros((length, length), device=device, dtype=dtype)
        if self.self_loop:
            A.fill_diagonal_(1.0)
        for k in range(1, self.neighbors + 1):
            if k < length:
                A[range(k, length), range(length - k)] = 1.0  # below diagonal
                A[range(length - k), range(k, length)] = 1.0  # above diagonal
        # Row-normalize
        rowsum = A.sum(dim=1, keepdim=True)
        A = A / rowsum.clamp_min(1e-20)

        self._cached_A = A
        self._cached_length = length
        return A

    def _average_neighbors_loop(
        self,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """Original loop-based implementation."""
        agg = torch.zeros_like(g)
        deg = torch.zeros((len(g), 1), device=g.device, dtype=g.dtype)
        if self.self_loop:
            agg += g
            deg += 1.0
        for k in range(1, self.neighbors + 1):
            if k < len(g):
                agg[k:] += g[:-k]
                deg[k:] += 1.0
                agg[:-k] += g[k:]
                deg[:-k] += 1.0
        avg = agg / deg.clamp_min(1e-20)
        return avg

    def forward(
        self,
        f: torch.Tensor,
    ) -> torch.Tensor:
        g = f.clone()
        if self.use_matrix:
            A = self._get_adjacency_matrix(len(f), f.device, f.dtype)
            for _ in range(self.num_steps):
                Ag = A @ g
                g = (1.0 - self.alpha) * f + self.alpha * Ag
        else:
            for _ in range(self.num_steps):
                Ag = self._average_neighbors_loop(g)
                g = (1.0 - self.alpha) * f + self.alpha * Ag
        return g

class ExactSm(Sm):
    def __init__(
        self, 
        alpha: float = 0.5, 
        learnable_alpha: bool = True, 
        neighbors: int = 1, 
        self_loop: bool = False,
    ):
        super().__init__(
            alpha = alpha, 
            learnable_alpha = learnable_alpha, 
            neighbors = neighbors, 
            self_loop = self_loop, 
        )

    def forward(
        self, 
        f: torch.Tensor, 
    ) -> torch.Tensor:
        A = self._adajency_matrix(len(f), device=f.device, dtype=f.dtype)
        M = torch.eye(len(f), device=f.device, dtype=f.dtype) - self.alpha * A
        rhs = (1.0 - self.alpha) * f
        g = torch.linalg.solve(M, rhs)
        return g

    def _adjacency_matrix(
        self,
        length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        A = torch.zeros((length, length), device=device, dtype=dtype)
        if self.self_loop:
            A.fill_diagonal_(1.0)
        for k in range(1, self.neighbors + 1):
            A[k:, :-k] += torch.eye(length - k, device=device, dtype=dtype)
            A[:-k, k:] += torch.eye(length - k, device=device, dtype=dtype)
        rowsum = A.sum(dim=1, keepdim=True)
        A = A / rowsum.clamp_min(1e-20)
        return A

class SmAP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        alpha: float = 0.5,
        num_steps: int = 10,
        variant: str = 'early',
        neighbors: int = 1,
    ):
        super().__init__()
        fc1 = torch.nn.Linear(in_features=in_features, out_features=hidden_dim)
        fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=1)
        assert variant in ['late', 'mid', 'early']
        self.variant = variant
        if self.variant == 'mid':
            fc2 = torch.nn.utils.parametrizations.spectral_norm(fc2)
        elif self.variant == 'early':
            fc1 = torch.nn.utils.parametrizations.spectral_norm(fc1)
            fc2 = torch.nn.utils.parametrizations.spectral_norm(fc2)
        self.mlp = torch.nn.Sequential(fc1, torch.nn.Tanh(), fc2)
        self.sm_layer = ApproxSm(alpha=alpha, learnable_alpha=True, num_steps=num_steps, neighbors=neighbors)

    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.variant == 'late':
            attn_logits = self.mlp(x)
            attn_logits = torch.cat([
                self.sm_layer(attn_logits_i)
                for attn_logits_i in torch.split(attn_logits, lengths)
            ])
        elif self.variant == 'mid':
            out = self.mlp[0](x)
            out = torch.cat([
                self.sm_layer(out_i)
                for out_i in torch.split(out, lengths)
            ])
            attn_logits = self.mlp[1:](out)
        elif self.variant == 'early':
            x = torch.cat([
                self.sm_layer(x_i)
                for x_i in torch.split(x, lengths)
            ])
            attn_logits = self.mlp(x)
        # Same as ABMIL after smooth operator
        attn_weights = torch.cat([
            torch.nn.functional.softmax(attn_logits_i, dim=0)
            for attn_logits_i in torch.split(attn_logits, lengths)
        ])
        out = attn_weights * x
        out = torch.cat([
            torch.sum(out_i, dim=0, keepdim=True)
            for out_i in torch.split(out, lengths)
        ])
        return out, attn_weights
<<<<<<< Updated upstream
    
=======
        
class SmTransformerLayer(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        num_heads: int = 8,
        alpha: float = 0.5,
        num_steps: int = 10,
        neighbors: int = 1,
    ):
        super().__init__()
        self.norm = torch.nn.LayerNorm(normalized_shape=in_features)
        self.attn = torch.nn.MultiheadAttention(embed_dim=in_features, num_heads=num_heads)
        self.sm_layer = ApproxSm(alpha=alpha, learnable_alpha=True, num_steps=num_steps, neighbors=neighbors)

    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        norm_x = self.norm(x)
        out, attn_weights = zip(*[
            self.attn(x_i, x_i, x_i, need_weights=True, average_attn_weights=False) 
            for x_i in torch.split(norm_x, lengths)
        ])
        out = [self.sm_layer(out_i) for out_i in out]
        out = x + torch.cat(out)
        return out, attn_weights
    
class SmTAP(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        num_heads: int = 8,
        alpha: float = 0.5,
        num_steps: int = 10,
        neighbors: int = 1,
    ):
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.randn(size=(1, in_features,)))
        self.layer1 = SmTransformerLayer(in_features, num_heads, alpha, num_steps, neighbors)
        self.pos_layer = PPEG(in_features=in_features)
        self.layer2 = SmTransformerLayer(in_features, num_heads, alpha, num_steps, neighbors)

    def forward(self, x, lengths):
        x = torch.cat([
            torch.cat((self.cls_token, x_i))
            for x_i in torch.split(x, lengths)
        ])
        lengths = tuple(length + 1 for length in lengths)
        out, _ = self.layer1(x, lengths)
        out = self.pos_layer(out, lengths)
        out, attn_weights = self.layer2(out, lengths)
        out = torch.stack([
            out_i[0,:]
            for out_i in torch.split(out, lengths)
        ])
        attn_weights = torch.cat([
            attn_weights_i[:,0,1:].T
            for attn_weights_i in attn_weights
        ])
        return out, attn_weights
    
>>>>>>> Stashed changes
