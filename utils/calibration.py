import torch
import torch.nn as nn
import torch.optim as optim

class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for post-hoc calibration of neural network logits.
    Rectifies Gap 6 (probability miscalibration under focal loss in PPIS).
    """
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        temperature = self.temperature.clamp(min=0.01)
        return logits / temperature

    def fit(self, val_logits, val_targets, max_iter=50, lr=0.01):
        """
        Fit temperature parameter using L-BFGS or Adam on validation set logits.
        """
        self.train()
        device = val_logits.device
        self.to(device)

        nll_criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits.view(-1), val_targets.float().view(-1))
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.eval()
        return self.temperature.item()
