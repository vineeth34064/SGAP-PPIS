import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TemperatureScaling(nn.Module):
    """
    Platt Temperature Scaling for post-hoc calibration of neural network logits.
    Rectifies Gap 6 (probability miscalibration under asymmetric focal loss in PPIS).
    Supports both binary 2-class logits (N, 2) and 1D logits (N, 1) or (N,).
    """
    def __init__(self, init_temp=1.5):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits):
        """Scale logits by learned temperature parameter."""
        temperature = self.temperature.clamp(min=0.01)
        return logits / temperature

    def calibrate_probs(self, logits):
        """Compute calibrated positive class probabilities."""
        scaled_logits = self.forward(logits)
        if scaled_logits.dim() == 2 and scaled_logits.size(1) == 2:
            return F.softmax(scaled_logits, dim=-1)[:, 1]
        else:
            return torch.sigmoid(scaled_logits.view(-1))

    def fit(self, val_logits, val_targets, max_iter=50, lr=0.01):
        """
        Fit temperature parameter using L-BFGS on validation set logits.
        Correctly handles both 2-class logits (N, 2) and 1D logits (N,) or (N, 1).
        """
        self.train()
        device = val_logits.device
        self.to(device)

        is_2class = (val_logits.dim() == 2 and val_logits.size(1) == 2)
        if is_2class:
            ce_criterion = nn.CrossEntropyLoss()
            targets = val_targets.long().view(-1)
        else:
            bce_criterion = nn.BCEWithLogitsLoss()
            targets = val_targets.float().view(-1)

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            if is_2class:
                loss = ce_criterion(scaled_logits, targets)
            else:
                loss = bce_criterion(scaled_logits.view(-1), targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.eval()
        return float(self.temperature.item())
