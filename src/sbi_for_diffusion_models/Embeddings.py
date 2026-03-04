import torch
import torch.nn as nn
from sbi.neural_nets.embedding_nets.fully_connected import FCEmbedding

class PermutationInvariantEmbedding(nn.Module):
    """
    Permutation-invariant session embedding
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

        self.num_trials = int(num_trials)
        self.trial_dim = int(trial_dim)
        self.aggregation = aggregation

        self.trial_net = FCEmbedding(
            input_dim=self.trial_dim,
            output_dim=trial_net_output_dim,
            num_layers=trial_net_layers,
            num_hiddens=trial_net_hidden,
        )

        layers: list[nn.Module] = []
        in_dim = int(trial_net_output_dim)
        for _ in range(int(post_agg_layers)):
            layers.append(nn.Linear(in_dim, int(post_agg_hidden)))
            layers.append(nn.ReLU())
            in_dim = int(post_agg_hidden)
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.post_net = nn.Sequential(*layers)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        B = x_flat.shape[0]

        # (B, T*D) -> (B, T, D)
        x_3d = x_flat.view(B, self.num_trials, self.trial_dim)

        # Embed each trial: FCEmbedding expects (B*T, D)
        emb = self.trial_net(x_3d.reshape(B * self.num_trials, self.trial_dim))
        emb = emb.view(B, self.num_trials, -1)  # (B, T, E)

        if self.aggregation == "mean":
            agg = emb.mean(dim=1)
        else:
            agg = emb.sum(dim=1)

        return self.post_net(agg)