import torch
import torch.nn as nn
from sbi.neural_nets.embedding_nets.fully_connected import FCEmbedding

class MaskAwarePermutationInvariantEmbedding(nn.Module):
    """
    Mask-aware permutation-invariant embedding -- handles 
    masking from data simulation (if trials go beyond T_MAX)

    trial_features: (B, T, D) where last dim includes mask as last channel.
    mask: (B, T, 1) in {0,1}
    """
    def __init__(
        self,
        *,
        num_trials: int,
        trial_dim: int,     
        trial_net_hidden: int = 128,
        trial_net_layers: int = 3,
        trial_net_output_dim: int = 64,
        post_agg_hidden: int = 128,
        post_agg_layers: int = 2,
        output_dim: int = 64,
        aggregation: str = "mean",  # "sum" or "mean"
    ):
        super().__init__()
        if aggregation not in ("sum", "mean"):
            raise ValueError("aggregation must be 'sum' or 'mean'")

        self.num_trials = num_trials
        self.trial_dim = trial_dim
        self.aggregation = aggregation

        # embed only the non-mask features.
        self.trial_feat_dim = trial_dim - 1

        self.trial_net = FCEmbedding(
            input_dim=self.trial_feat_dim,
            output_dim=trial_net_output_dim,
            num_layers=trial_net_layers,
            num_hiddens=trial_net_hidden,
        )

        # Post-aggregation network
        layers = []
        in_dim = trial_net_output_dim
        for _ in range(post_agg_layers):
            layers += [nn.Linear(in_dim, post_agg_hidden), nn.ReLU()]
            in_dim = post_agg_hidden
        layers += [nn.Linear(in_dim, output_dim)]
        self.post_net = nn.Sequential(*layers)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        B = x_flat.shape[0]
        x_3d = x_flat.view(B, self.num_trials, self.trial_dim)

        feats = x_3d[..., : self.trial_feat_dim] # (B,T,D-1)
        mask = x_3d[..., self.trial_feat_dim :]  # (B,T,1)

        # Embed each trial
        emb = self.trial_net(feats.reshape(B * self.num_trials, self.trial_feat_dim))
        emb = emb.view(B, self.num_trials, -1)            # (B,T,E)
        emb = emb * mask  # Apply mask            

        summed = emb.sum(dim=1)                            # (B,E)
        if self.aggregation == "mean":
            denom = mask.sum(dim=1).clamp_min(1.0)        # (B,1)
            agg = summed / denom
        else:
            agg = summed

        return self.post_net(agg)