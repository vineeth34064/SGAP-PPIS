import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for binary residue classification in PPIS as defined in Section II-G:
    FL(p_t) = - Phi_t * (1 - p_t)^gamma * log(p_t)
    where:
      - p_t = p if y=1 else (1 - p)
      - Phi_t = 0.75 if y=1 else 0.25
      - gamma = 2.0
    """
    def __init__(self, alpha_pos=0.75, alpha_neg=0.25, gamma=2.0, reduction='mean', eps=1e-8):
        super(FocalLoss, self).__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets, rsa=None, surface_penalty_weight=0.0):
        """
        Args:
            logits: (N, 2) or (N, 1) or (N,) unnormalized logits
            targets: (N,) binary ground truth {0, 1}
            rsa: (N,) Relative Solvent Accessibility from DSSP (optional, for Gap 7 rectification)
            surface_penalty_weight: float weight for penalizing buried false-positive interaction predictions
        """
        targets = targets.float().view(-1)
        if logits.dim() == 2 and logits.size(1) == 2:
            # 2-class cross-entropy style
            probs = F.softmax(logits, dim=1)
            p_pos = probs[:, 1].clamp(self.eps, 1.0 - self.eps)
        else:
            p_pos = torch.sigmoid(logits.view(-1)).clamp(self.eps, 1.0 - self.eps)

        p_t = torch.where(targets == 1, p_pos, 1.0 - p_pos)
        phi_t = torch.where(targets == 1, self.alpha_pos, self.alpha_neg)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        loss = - phi_t * modulating_factor * torch.log(p_t)

        # Gap 7 Rectification: Surface Feasibility Penalty
        # Buried non-binding residues (RSA < 0.05 and targets == 0) predicting high interaction probability
        # receive an auxiliary biophysical consistency penalty.
        # Conditioned on (1.0 - targets) to prevent destructive gradient conflict on true active sites in deep pockets.
        # Guarded by torch.any(rsa > 0.0) to avoid false penalty if DSSP file was absent.
        if rsa is not None and surface_penalty_weight > 0.0:
            rsa = rsa.view(-1)
            if torch.any(rsa > 0.0):
                buried_non_binding = (rsa < 0.05).float() * (1.0 - targets)
                surface_penalty = buried_non_binding * (p_pos ** 2)
                loss = loss + surface_penalty_weight * surface_penalty

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
