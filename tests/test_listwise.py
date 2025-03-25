import torch

from ltr.listwise import ListwiseLTR


def test_listwise_ltr_training():
    torch.manual_seed(42)
    model = ListwiseLTR(input_dim=2)

    x = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
    y = torch.tensor([2.0, 1.0, 3.0])  # relevância

    initial_loss = model.train_step(x, y)
    for _ in range(10):
        loss = model.train_step(x, y)
    assert loss < initial_loss
