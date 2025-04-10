import torch
import torch.nn as nn
import torch.optim as optim


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.zeros(input_dim))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.matmul(x, self.weights) + self.bias

    def predict(self, test_loader):
        self.eval()
        y_pred = []
        y_true = []

        device = next(self.parameters()).device

        with torch.no_grad():
            for (x_batch, y_batch) in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                logits = self(x_batch)

                probabilities = torch.sigmoid(x_batch)

                predictions = (probabilities >= 0.5).float()

                y_pred.extend(predictions.cpu().numpy())
                y_true.extend(y_batch.cpu().numpy())
                   
        return torch.tensor(y_true, dtype=torch.float32), torch.tensor(y_pred, dtype=torch.float32)