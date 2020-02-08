import torch

class AntiSpecificityLoss(torch.nn.Module):
    """
    Specificity Loss = (False Positive) / (FP + TN)

    """
    def __init__(self, scale=1):
        super(AntiSpecificityLoss, self).__init__()
        self.scale = scale

    def forward(self, pred_edge, no_edge):
        FP = pred_edge * no_edge
        TN = (1-pred_edge) * no_edge
        TN_viewed = TN.view(TN.size(0), -1).mean(1).view(-1, 1, 1, 1)
        FP_viewed = FP.view(FP.size(0), -1).mean(1).view(-1, 1, 1, 1)
        eps = 1e-7
        return self.scale*torch.mean(FP_viewed / (TN_viewed + FP_viewed + eps))