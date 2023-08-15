import torch
import torch.nn.functional as F

def gramschmidt(v, epsilon=0):
    """
    Args:
        v (..., 6 tensor): batch of 6D vectors
    Returns:
        batch of rotation matrices (..., 3, 3 tensor)
    """
    x = v[..., :3]
    y = v[..., 3:6]

    x = F.normalize(x, dim=-1)
    y = F.normalize(y - x * torch.sum(x * y, dim=-1, keepdim=True), dim=-1)
    z = torch.cross(x, y)
    R = torch.stack([x, y, z], axis=-1)
    
    return R

def xy_loss(R_pred, R_gt):
    """
    Args:
        R_pred (..., 3, 3 tensor): batch of predicted rotation matrices
        R_gt (..., 3, 3 tensor): batch of ground truth rotation matrices
    Returns:
        batch of xy loss (..., tensor)
    """
    losses = {}
    losses['x_dir'] = (1 - (R_pred[..., 0] * R_gt[..., 0]).sum(dim=1)).mean()
    losses['y_dir'] = (1 - (R_pred[..., 1] * R_gt[..., 1]).sum(dim=1)).mean()
    losses['loss'] = losses['x_dir'] + losses['y_dir']
    return losses
