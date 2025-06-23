import torch
import torch.nn.functional as F
from typing import Dict, List

class GumbelSoftmaxSampler:
    """
    Updated Gumbel-Softmax sampling using marginal probabilities.
    """
    
    @staticmethod
    def sample_gumbel(shape: torch.Size, device: str = 'cpu') -> torch.Tensor:
        """Sample from Gumbel(0,1) distribution."""
        uniform = torch.rand(shape, device=device)
        return -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
    
    @staticmethod
    def gumbel_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Apply Gumbel-Softmax sampling."""
        gumbel_noise = GumbelSoftmaxSampler.sample_gumbel(logits.shape, device=logits.device)
        return F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    
    def sample_hidden_links_batch(self, 
                                 marginal_probs: Dict[str, torch.Tensor],
                                 temperature: float,
                                 num_samples: int = 1) -> List[Dict[str, torch.Tensor]]:
        """
        Sample hidden links using marginal probabilities.
    
        """
        all_samples = []
        
        for sample_idx in range(num_samples):
            samples = {}
            for pair_key, marginal_prob in marginal_probs.items():
                # Use marginal probabilities as logits
                marginal_logits = torch.log(marginal_prob + 1e-8)
                samples[pair_key] = self.gumbel_softmax(marginal_logits, temperature)
            all_samples.append(samples)
        
        return all_samples