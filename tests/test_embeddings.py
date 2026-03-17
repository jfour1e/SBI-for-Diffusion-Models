import pytest
import torch

from sbi_for_diffusion_models.Embeddings import MaskAwarePermutationInvariantEmbedding


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mask_aware_embedding_padding_invariance(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)

    T = 8
    P = 5
    trial_dim = 2 + P + 1  # last is mask
    x_dim = T * trial_dim

    net = MaskAwarePermutationInvariantEmbedding(
        num_trials=T,
        trial_dim=trial_dim,
        trial_net_hidden=16,
        trial_net_layers=2,
        trial_net_output_dim=8,
        post_agg_hidden=16,
        post_agg_layers=1,
        output_dim=8,
        aggregation="mean",
    ).to(dev)

    # Build a batch of 1 with 3 valid trials and 5 padded
    x_3d = torch.zeros((1, T, trial_dim), device=dev)

    # Fill 3 valid trials with arbitrary numbers; set mask=1
    x_3d[0, 0, :2] = torch.tensor([0.2, 1.0], device=dev)  # rt, choice
    x_3d[0, 0, 2:2+P] = 1.0
    x_3d[0, 0, -1] = 1.0

    x_3d[0, 1, :2] = torch.tensor([0.3, 0.0], device=dev)
    x_3d[0, 1, 2:2+P] = -1.0
    x_3d[0, 1, -1] = 1.0

    x_3d[0, 2, :2] = torch.tensor([0.4, 1.0], device=dev)
    x_3d[0, 2, 2:2+P] = 1.0
    x_3d[0, 2, -1] = 1.0

    x_flat = x_3d.view(1, -1)

    # Create an alternative version with different padded garbage but mask=0 there
    x_3d_alt = x_3d.clone()
    x_3d_alt[0, 3:, :-1] = torch.randn((T-3, trial_dim-1), device=dev)  # garbage
    x_3d_alt[0, 3:, -1] = 0.0  # mask zero ensures ignored
    x_flat_alt = x_3d_alt.view(1, -1)

    y = net(x_flat)
    y_alt = net(x_flat_alt)

    # Must be identical (or extremely close) since masked trials should not matter
    torch.testing.assert_close(y, y_alt, rtol=0, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mask_aware_embedding_permutation_invariance(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)

    T = 6
    P = 3
    trial_dim = 2 + P + 1
    net = MaskAwarePermutationInvariantEmbedding(
        num_trials=T,
        trial_dim=trial_dim,
        trial_net_hidden=16,
        trial_net_layers=2,
        trial_net_output_dim=8,
        post_agg_hidden=16,
        post_agg_layers=1,
        output_dim=8,
        aggregation="mean",
    ).to(dev)

    # Make 6 valid trials, then permute their order
    x_3d = torch.zeros((1, T, trial_dim), device=dev)
    for t in range(T):
        x_3d[0, t, 0] = 0.1 * (t + 1)          # rt
        x_3d[0, t, 1] = float(t % 2)           # choice
        x_3d[0, t, 2:2+P] = 1.0 if (t % 2) else -1.0
        x_3d[0, t, -1] = 1.0                   # mask

    perm = torch.randperm(T, device=dev)
    x_perm = x_3d[:, perm, :]

    y = net(x_3d.view(1, -1))
    y_perm = net(x_perm.view(1, -1))

    torch.testing.assert_close(y, y_perm, rtol=0, atol=1e-6)