import torch

def apply_cmvn(feats):
    feats = feats - feats.mean(1, keepdims=True)
    feats = feats / feats.std(1, keepdims=True)
    return feats

class Snake(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor):
        return x + (1/self.alpha) * torch.sin(self.alpha * x)**2