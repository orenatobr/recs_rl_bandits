import torch

from ltr.pairwise import PairwiseLTR


def test_pairwise_ltr_training():
    torch.manual_seed(42)
    model = PairwiseLTR(input_dim=2)

    x_i = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
    x_j = torch.tensor([[0.2, 0.8], [0.1, 0.9]])
    y_ij = torch.tensor([1, 1])  # i é preferido a j

    initial_loss = model.train_step(x_i, x_j, y_ij)
    for _ in range(10):
        loss = model.train_step(x_i, x_j, y_ij)
    assert loss < initial_loss
