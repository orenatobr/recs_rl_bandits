import torch.nn as nn
import torch.optim as optim


class PairwiseLTR(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def train_step(self, x_i, x_j, y_ij):
        self.optimizer.zero_grad()
        s_i = self.forward(x_i)
        s_j = self.forward(x_j)
        preds = s_i - s_j
        loss = self.loss_fn(preds, y_ij.float())
        loss.backward()
        self.optimizer.step()
        return loss.item()
