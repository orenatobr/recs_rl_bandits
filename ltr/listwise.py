import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ListwiseLTR(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def softmax_loss(self, scores, labels):
        scores_prob = F.log_softmax(scores, dim=0)
        labels_prob = F.softmax(labels, dim=0)
        return F.kl_div(scores_prob, labels_prob, reduction="batchmean")

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        scores = self.forward(x)
        loss = self.softmax_loss(scores, y.float())
        loss.backward()
        self.optimizer.step()
        return loss.item()
